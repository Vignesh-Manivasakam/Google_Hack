#!/usr/bin/env python3
"""
Live + Saved Layer Activation Visualizer for Gemma 4 E4B.

Single-file diagnostic tool for mqg1cob:
- Loads a submission folder once (supports submission_E4B_LoRA_v1, v3_ArmB, etc.).
- Runs one task (--task_id) or a batch (--task_ids a,b,c) using the full tasks.jsonl pool
  so holdout30/train99/smoke4 tasks all resolve accurately without silent fallback.
- Hooks all 42 text decoder layers, recording per-step ||h_out - h_in|| / ||h_in||
  split into thought vs action tokens AND grouped by tool type (edit_file, read_file,
  search_similar_code, run_command, submit_patch).
- Saves non-overwriting outputs under outputs/<submission_name>/<task_id>/ including
  full 42-layer vectors, top-8 tool-specialized, top-8 reasoning-specialized, and episode
  outcome (patch_bytes, edit_oks, edit_fails, submitted).
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONUTF8"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GEMMA_DIR = Path(r"C:\Users\mqg1cob\Documents\Google Hack\gemma-4-developer-agent")
DEFAULT_SUBMISSION_CANDIDATES = [
    DEFAULT_GEMMA_DIR / "submission_E4B_LoRA_v1",
    Path(r"C:\Users\mqg1cob\OneDrive - Bosch Group\Hackathon\submission(1)"),
]
DEFAULT_MODEL_CANDIDATES = [
    Path(r"C:\Users\mqg1cob\Documents\Google Hack\gemma-4-e4b-it"),
    Path(r"C:\Users\mqg1cob\OneDrive - Bosch Group\Hackathon\Google Hack\gemma-4-e4b-it"),
]


def resolve_default_submission() -> Path:
    for p in DEFAULT_SUBMISSION_CANDIDATES:
        if p.exists():
            return p
    return DEFAULT_SUBMISSION_CANDIDATES[0]


def resolve_default_model() -> str:
    for p in DEFAULT_MODEL_CANDIDATES:
        if p.exists():
            return str(p)
    return str(DEFAULT_MODEL_CANDIDATES[0])


def load_submission_config(sub_dir: Path) -> dict:
    sub_dir = sub_dir.resolve()
    prompt_path = sub_dir / "prompts" / "system.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing {prompt_path}")
    prompt_md = prompt_path.read_text(encoding="utf-8")

    temp, top_p, thinking_budget = 0.15, 0.90, 1024
    samp_path = sub_dir / "configs" / "sampling.yaml"
    if samp_path.exists():
        stxt = samp_path.read_text(encoding="utf-8")
        m_t = re.search(r"temperature:\s*([0-9.]+)", stxt)
        m_p = re.search(r"top_p:\s*([0-9.]+)", stxt)
        m_tb = re.search(r"thinking_budget:\s*([0-9]+)", stxt)
        if m_t:
            temp = float(m_t.group(1))
        if m_p:
            top_p = float(m_p.group(1))
        if m_tb:
            thinking_budget = int(m_tb.group(1))

    max_steps = 40
    eval_path = sub_dir / "eval_config.yaml"
    if eval_path.exists():
        etxt = eval_path.read_text(encoding="utf-8")
        m_ms = re.search(r"max_tool_calls:\s*([0-9]+)", etxt)
        if m_ms:
            max_steps = int(m_ms.group(1))

    adapter_dir = None
    use_analyzer = False
    agent_path = sub_dir / "agent.yaml"
    if agent_path.exists():
        atxt = agent_path.read_text(encoding="utf-8")
        m_ad = re.search(r"^adapter:\s*(\S+)", atxt, re.MULTILINE)
        if m_ad:
            cand_ad = sub_dir / "adapters" / m_ad.group(1).strip()
            if cand_ad.exists():
                adapter_dir = cand_ad
        if "code_analyzer" in atxt:
            use_analyzer = True

    return {
        "prompt_md": prompt_md,
        "temp": temp,
        "top_p": top_p,
        "thinking_budget": thinking_budget,
        "max_steps": max_steps,
        "adapter_dir": adapter_dir,
        "use_analyzer": use_analyzer,
    }


def find_decoder_layers(model):
    """Auto-discover text decoder ModuleList across Gemma 4, multimodal, and PEFT wrappers."""
    matches = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and (name == "layers" or name.endswith(".layers")):
            low = name.lower()
            if "vision" in low or "audio" in low or "encoder" in low:
                continue
            matches.append((name, mod))

    if matches:
        matches.sort(key=lambda x: ("language_model" in x[0].lower(), len(x[1])), reverse=True)
        chosen_name, chosen_mod = matches[0]
        print(f"[HOOKS] Found {len(chosen_mod)} decoder layers at: {chosen_name}")
        return chosen_mod

    all_ml = [(n, len(m)) for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList)]
    raise RuntimeError(f"Could not locate decoder layers. Available ModuleLists: {all_ml}")


class LiveLayerVisualizer:
    """Hooks all decoder layers, updates a live dashboard each step, and saves PNG/JSON."""

    def __init__(self, model, task_id: str, out_dir: Path, live: bool = True):
        self.layers = find_decoder_layers(model)
        self.num_layers = len(self.layers)
        self.task_id = task_id
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.live = live
        self.hooks = []
        self._active = False
        self._cur_token_vec = None
        self._turn_tokens = []
        self.turn_records = []

        self.fig = None
        self.axes = None
        if self.live:
            try:
                plt.ion()
                self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 6))
                self.fig.canvas.manager.set_window_title(f"LIVE Gemma 4 E4B Layers — {task_id}")
                plt.show(block=False)
                plt.pause(0.05)
            except Exception as e:
                print(f"[INFO] Live GUI window unavailable ({e}); will save images after each step.")
                self.live = False

        for idx, layer in enumerate(self.layers):
            self.hooks.append(layer.register_forward_hook(self._make_hook(idx), with_kwargs=True))

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs, outputs):
            if not self._active:
                return
            h_in = args[0] if args else kwargs.get("hidden_states")
            if h_in is None or not torch.is_tensor(h_in):
                return
            if h_in.ndim != 3 or h_in.shape[1] != 1:
                return
            h_out = outputs[0] if isinstance(outputs, tuple) else outputs
            if not torch.is_tensor(h_out):
                return
            h_in_f = h_in[:, -1, :].float()
            h_out_f = h_out[:, -1, :].float()
            delta = (torch.norm(h_out_f - h_in_f, dim=-1) / (torch.norm(h_in_f, dim=-1) + 1e-6)).mean().item()

            if layer_idx == 0:
                self._cur_token_vec = np.zeros(self.num_layers, dtype=np.float32)
            if self._cur_token_vec is not None:
                self._cur_token_vec[layer_idx] = delta
                if layer_idx == self.num_layers - 1:
                    self._turn_tokens.append(self._cur_token_vec.copy())
        return hook

    def start_turn(self):
        self._turn_tokens = []
        self._cur_token_vec = None
        self._active = True

    def end_turn(self, step_idx: int, raw_txt: str, tool_name: str):
        self._active = False
        if not self._turn_tokens:
            return
        mat = np.stack(self._turn_tokens, axis=1)  # [num_layers, num_gen_tokens]
        n_tok = mat.shape[1]

        thought_ratio = 0.0
        if "<channel|>" in raw_txt:
            pre, _ = raw_txt.split("<channel|>", 1)
            thought_ratio = len(pre) / max(1, len(raw_txt))
        split_idx = int(round(n_tok * thought_ratio))

        mean_all = mat.mean(axis=1)
        mean_thought = mat[:, :split_idx].mean(axis=1) if split_idx > 0 else np.zeros(self.num_layers, dtype=np.float32)
        mean_action = mat[:, split_idx:].mean(axis=1) if split_idx < n_tok else mean_all

        rec = {
            "step": step_idx,
            "tool": tool_name or "no_tool",
            "num_gen_tokens": n_tok,
            "mean_all": mean_all,
            "mean_thought": mean_thought,
            "mean_action": mean_action,
            "has_thought": split_idx > 0,
        }
        self.turn_records.append(rec)

        top5 = np.argsort(mean_all)[::-1][:5].tolist()
        top5_str = ", ".join(f"L{i}({mean_all[i]:.2f})" for i in top5)
        print(f"  [LIVE STEP {step_idx:02d}] tool={rec['tool']:<20} | tokens={n_tok:<4} | hottest: {top5_str}", flush=True)
        self._refresh_plots(save_final=False)

    def _refresh_plots(self, save_final: bool = False, outcome: dict | None = None):
        recs = self.turn_records
        if not recs:
            return

        heatmap = np.stack([r["mean_all"] for r in recs], axis=1)
        labels = [f"s{r['step']:02d}:{r['tool']}" for r in recs]
        layers_x = np.arange(self.num_layers)

        thought_vecs = [r["mean_thought"] for r in recs if r["has_thought"]]
        action_vecs = [r["mean_action"] for r in recs if r["tool"] != "no_tool"]
        mean_thought = np.mean(thought_vecs, axis=0) if thought_vecs else np.zeros(self.num_layers)
        mean_action = np.mean(action_vecs, axis=0) if action_vecs else heatmap.mean(axis=1)
        diff = mean_action - mean_thought

        if self.live and self.fig is not None:
            ax1, ax2 = self.axes
            ax1.clear()
            ax1.imshow(heatmap, aspect="auto", origin="lower", cmap="inferno")
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax1.set_yticks(range(0, self.num_layers, 3))
            ax1.set_xlabel("Step & Tool")
            ax1.set_ylabel(f"Layer Index (0-{self.num_layers - 1})")
            ax1.set_title(f"LIVE Per-Step Layer Activity ({self.task_id})")

            ax2.clear()
            ax2.plot(layers_x, mean_thought, marker="o", markersize=4, label="Reasoning (thought)", color="#4C72B0")
            ax2.plot(layers_x, mean_action, marker="s", markersize=4, label="Tool-Calling (action)", color="#DD8452")
            ax2.set_xlabel(f"Layer Index (0-{self.num_layers - 1})")
            ax2.set_ylabel("Mean Relative Residual Delta")
            ax2.set_title("Reasoning vs. Tool-Calling Layers")
            ax2.legend()
            ax2.grid(alpha=0.3)

            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.05)
            self.fig.savefig(self.out_dir / "live_dashboard.png", dpi=130)

        if save_final:
            fig1, ax = plt.subplots(figsize=(max(9, len(recs) * 0.8), 7))
            im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="inferno")
            fig1.colorbar(im, ax=ax, label="Relative Residual Delta ||h_out - h_in|| / ||h_in||")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=9)
            ax.set_yticks(range(0, self.num_layers, 2))
            ax.set_xlabel("Agent Step & Tool Called")
            ax.set_ylabel("Gemma 4 E4B Layer Index")
            ax.set_title(f"Step-by-Layer Activation Map — {self.task_id}")
            fig1.tight_layout()
            p1 = self.out_dir / "step_by_layer_heatmap.png"
            fig1.savefig(p1, dpi=150)
            plt.close(fig1)

            fig2, (a0, a1) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
            a0.plot(layers_x, mean_thought, marker="o", label="Reasoning (thought tokens)", color="#4C72B0")
            a0.plot(layers_x, mean_action, marker="s", label="Tool-Calling (action tokens)", color="#DD8452")
            a0.set_ylabel("Mean Relative Delta")
            a0.set_title(f"Reasoning vs. Tool-Calling Layer Profile — {self.task_id}")
            a0.legend()
            a0.grid(alpha=0.3)

            colors = ["#C44E52" if v > 0 else "#4C72B0" for v in diff]
            a1.bar(layers_x, diff, color=colors)
            a1.axhline(0, color="black", linewidth=0.8)
            a1.set_xlabel(f"Gemma 4 E4B Layer Index (0..{self.num_layers - 1})")
            a1.set_ylabel("Tool-Call minus Reasoning")
            a1.set_title("Contrastive Score (Red = Tool-Specialized, Blue = Reasoning-Specialized)")
            a1.set_xticks(range(0, self.num_layers, 2))
            a1.grid(alpha=0.3)
            fig2.tight_layout()
            p2 = self.out_dir / "tool_vs_thought_profile.png"
            fig2.savefig(p2, dpi=150)
            plt.close(fig2)

            top_tool = sorted(np.argsort(diff)[::-1][:8].tolist())
            top_reason = sorted(np.argsort(diff)[:8].tolist())
            top_action = sorted(np.argsort(mean_action)[::-1][:8].tolist())

            by_tool = {}
            for tname in sorted({r["tool"] for r in recs}):
                vecs = [r["mean_all"] for r in recs if r["tool"] == tname]
                by_tool[tname] = {
                    "count": len(vecs),
                    "mean_by_layer": [round(float(x), 5) for x in np.mean(vecs, axis=0)],
                }

            summary = {
                "task_id": self.task_id,
                "num_layers": self.num_layers,
                "num_steps_recorded": len(recs),
                "outcome": outcome or {},
                "top8_tool_specialized_layers": top_tool,
                "top8_reasoning_specialized_layers": top_reason,
                "top8_highest_action_layers": top_action,
                "mean_thought_by_layer": [round(float(x), 5) for x in mean_thought],
                "mean_action_by_layer": [round(float(x), 5) for x in mean_action],
                "diff_action_minus_thought_by_layer": [round(float(x), 5) for x in diff],
                "by_tool": by_tool,
                "steps": [
                    {
                        "step": r["step"],
                        "tool": r["tool"],
                        "tokens": r["num_gen_tokens"],
                        "top5_layers": np.argsort(r["mean_all"])[::-1][:5].tolist(),
                    }
                    for r in recs
                ],
            }
            p3 = self.out_dir / "layer_summary.json"
            p3.write_text(json.dumps(summary, indent=2), encoding="utf-8")

            print(f"\n[DONE] Saved outputs inside: {self.out_dir}")
            print(f"  - Top-8 Tool-Specialized Layers:      {top_tool}")
            print(f"  - Top-8 Reasoning-Specialized Layers: {top_reason}")
            print(f"  - Top-8 Highest-Action Layers:        {top_action}")

    def finish(self, outcome: dict | None = None):
        self._refresh_plots(save_final=True, outcome=outcome)
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        if self.live and self.fig is not None:
            plt.ioff()
            plt.show()


def main():
    ap = argparse.ArgumentParser(description="Live & Saved Layer Activation Visualizer for Gemma 4 E4B")
    ap.add_argument("--submission_dir", type=Path, default=resolve_default_submission())
    ap.add_argument("--gemma_dir", type=Path, default=DEFAULT_GEMMA_DIR)
    ap.add_argument("--model_id", type=str, default=resolve_default_model())
    ap.add_argument("--venv_root", type=Path, default=None)
    ap.add_argument("--task_mode", type=str, default="smoke4", choices=["smoke4", "smoke7", "val20", "shadow_dev", "holdout30", "all"])
    ap.add_argument("--task_index", type=int, default=0, help="Pick 0, 1, 2... when --task_id/--task_ids not set")
    ap.add_argument("--task_id", type=str, default="", help="Single exact instance_id (searches full tasks.jsonl)")
    ap.add_argument("--task_ids", type=str, default="", help="Comma-separated instance_ids to run sequentially on 1 GPU load")
    ap.add_argument("--max_steps", type=int, default=6, help="Step cap (e.g. 6 for fast clue, 40 for full solve)")
    ap.add_argument("--no_live", action="store_true", help="Disable live GUI window (save PNGs/JSON only)")
    args = ap.parse_args()

    gemma_dir = args.gemma_dir.resolve()
    venv_root = (args.venv_root or (gemma_dir.parent / ".win_venvs")).resolve()

    sys.path.insert(0, str(gemma_dir))
    from win_local_sandbox import ensure_repo_venv, extract_tgz_windows_safe, load_tasks_select
    from win_phase2_tournament import RepoCodeGraph, _build_workspace_tree
    from win_harness_triage_parallel import BatchedGemmaServer
    from win_frontier_6harness_overnight import (
        FrontierCandidate,
        _clear_shared_venv_pth,
        _deep_strip_multimodal_vram,
        _find_snapshot_file,
        _force_remove_readonly,
        run_frontier_agent,
    )

    sub_dir = args.submission_dir.resolve()
    sub_cfg = load_submission_config(sub_dir)
    all_tasks = load_tasks_select(gemma_dir / "tasks.jsonl", "all")
    by_id = {t["instance_id"]: t for t in all_tasks}

    if args.task_ids.strip():
        req_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]
        selected_tasks = []
        for tid in req_ids:
            if tid not in by_id:
                raise ValueError(f"Task ID '{tid}' not found in {gemma_dir / 'tasks.jsonl'}")
            selected_tasks.append(by_id[tid])
    elif args.task_id.strip():
        tid = args.task_id.strip()
        if tid not in by_id:
            raise ValueError(f"Task ID '{tid}' not found in {gemma_dir / 'tasks.jsonl'}")
        selected_tasks = [by_id[tid]]
    else:
        mode_tasks = load_tasks_select(gemma_dir / "tasks.jsonl", args.task_mode)
        selected_tasks = [mode_tasks[min(args.task_index, len(mode_tasks) - 1)]]

    print(f"[CONFIG] Submission: {sub_dir}")
    print(f"[CONFIG] Model:      {args.model_id}")
    print(f"[CONFIG] Adapter:    {sub_cfg['adapter_dir']}")
    print(f"[TASKS]  {len(selected_tasks)} task(s): {[t['instance_id'] for t in selected_tasks]} | max_steps={args.max_steps}")

    server = BatchedGemmaServer(model_id=args.model_id, max_batch_size=1, adapter_dir=sub_cfg["adapter_dir"])
    used_gb = _deep_strip_multimodal_vram(server)
    if sub_cfg["adapter_dir"] and not getattr(server.engine, "_peft_mounted", False):
        from peft import PeftModel
        if not isinstance(server.engine.model, PeftModel):
            server.engine.model = PeftModel.from_pretrained(server.engine.model, str(sub_cfg["adapter_dir"]))
            server.engine.model.eval()
            server.engine._peft_mounted = True
    print(f"[GPU] Ready ({used_gb:.2f} GiB VRAM).")

    orig_generate_turn = server.generate_turn
    sub_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", sub_dir.name)

    for task in selected_tasks:
        tid, repo = task["instance_id"], task["repo"]
        out_dir = SCRIPT_DIR / "outputs" / sub_tag / tid
        print(f"\n=== RUNNING {tid} ({repo}) -> {out_dir} ===")
        viz = LiveLayerVisualizer(server.engine.model, task_id=tid, out_dir=out_dir, live=not args.no_live)
        turn_counter = {"n": 0}

        def hooked_generate_turn(messages, tools=None, temperature=0.1, thinking_budget=1024, _viz=viz, _tc=turn_counter):
            _tc["n"] += 1
            _viz.start_turn()
            raw_txt, calls = orig_generate_turn(messages, tools=tools, temperature=temperature, thinking_budget=thinking_budget)
            tool_name = calls[0]["name"] if calls else "no_tool"
            _viz.end_turn(_tc["n"], raw_txt, tool_name)
            return raw_txt, calls

        server.generate_turn = hooked_generate_turn

        cfg = FrontierCandidate(
            cid="model_check_live",
            prompt_md=sub_cfg["prompt_md"],
            temp=sub_cfg["temp"],
            thinking_budget=sub_cfg["thinking_budget"],
            max_steps=min(args.max_steps, sub_cfg["max_steps"]),
            use_analyzer_tool=sub_cfg["use_analyzer"],
        )
        py_exe = ensure_repo_venv(venv_root, repo, wheels_dir=gemma_dir / "wheels").resolve()
        code_graph = RepoCodeGraph(gemma_dir, repo, task.get("base_commit", "HEAD"))
        bstate = {
            "tid": tid, "steps": 0, "edit_oks": 0, "edit_fails": 0, "malformed": 0,
            "no_call_turns": 0, "ping_pong_errors": 0, "no_call_samples": [],
            "submitted": False, "used_semantic": 0, "used_graph": 0, "trace": [], "messages": [],
        }

        snap_file = _find_snapshot_file(gemma_dir, tid)
        tmp_s = tempfile.mkdtemp(prefix=f"mcheck_{tid}_")
        outcome = {}
        try:
            ws = extract_tgz_windows_safe(snap_file, Path(tmp_s) / "workspace")
            if hasattr(code_graph, "index_workspace"):
                code_graph.index_workspace(ws)
            subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=str(ws), capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=str(ws), capture_output=True)
            subprocess.run(["git", "commit", "-m", "baseline", "--allow-empty", "-q"], cwd=str(ws), capture_output=True)
            _clear_shared_venv_pth(py_exe)
            ws_tree = _build_workspace_tree(ws)
            issue_prompt = f"Repository: {repo}\n\nProblem Description:\n{task['problem_statement'][:3200]}\n\nWorkspace Layout:\n{ws_tree}"
            run_frontier_agent(server, ws, py_exe, code_graph, cfg, issue_prompt, bstate)
            diff_res = subprocess.run(["git", "diff"], cwd=str(ws), capture_output=True, text=True)
            patch_bytes = len((diff_res.stdout or "").strip().encode("utf-8"))
            outcome = {
                "steps": bstate["steps"],
                "edit_oks": bstate["edit_oks"],
                "edit_fails": bstate["edit_fails"],
                "no_call_turns": bstate["no_call_turns"],
                "submitted": bstate["submitted"],
                "patch_bytes": patch_bytes,
            }
        finally:
            shutil.rmtree(tmp_s, onerror=_force_remove_readonly)

        viz.finish(outcome=outcome)


if __name__ == "__main__":
    main()

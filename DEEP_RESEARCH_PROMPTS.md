# Deep Research Prompt Pack — Kaggle "Gemma 4 Developer Agent" Final Sprint

**Prepared:** Nov 23, 2026 · **Sprint window:** Nov 23 – Dec 2, 2026 (final deadline 23:59 UTC)
**Scope:** 6 ready-to-paste deep-research prompts for external AI tools (ChatGPT Deep Research, Gemini Deep Research, Perplexity, Claude with web search), ranked by priority, each mapped to the exact decision and roadmap day it feeds.

---

## 0. My honest opinion first — where research helps, where it does NOT

Deep research = **external knowledge** (papers, docs, forums, GitHub issues). Your repo = **internal evidence** (your own experiments). Only send out questions where the missing knowledge lives *outside* your repo. Applying that filter to this project:

**Genuinely worth researching (in priority order):**

| # | Topic | Priority | Why it pays off | Feeds which decision |
|:--|:--|:--:|:--|:--|
| R1 | Small-model agentic SFT/LoRA science | **P0** | Your Slot 2 win condition ($10k, rivals tied at 0.10) rides on LoRA v2/v3 quality. Corpus design, curriculum, DPO/DAgger, TIES/DARE are literature-rich topics you have NOT yet tested. | Day 4 LoRA v2 training config; Day 6 TIES/DARE merge |
| R2 | Kaggle competition & leaderboard intelligence | **P0** | Top 3 sit at 0.29 / 0.248 / 0.242 while your B4 = 0.12 and B8 targets only 0.16–0.20. You are flying blind on what the top teams actually do. Also: public-LB task mix, seniority tie-break tactics, selection rules edge cases. | B8/B9 targeting; whether to pivot quota to Slot 2 (Nov 25 circuit breaker) |
| R3 | SWE-agent harness & prompt SOTA (2024–2026) | **P1** | Your B7→B8 RCA was purely internal. The SWE-bench community has published localization, test-selection, context-management and small-context-prompt techniques you can import into B8/B9 and the E4B prompt. | Day 3 B8 authoring; E4B prompt distillation (497→~100 tok) |
| R4 | vLLM + Compressed-Tensors QAT + PEFT serving stack | **P1** | You already hit one silent version landmine (torchao 0.10 vs PEFT). The exact serving flags, LoRA-over-quantized-checkpoint support matrix, and known vLLM bugs are version-specific external knowledge. | Master Engine Phase 3/4 correctness; any 31B serving change |
| R5 | Gemma architecture ground-truth verification | **P2** | Your layer targeting ([23,29,30,35,41]) and the 512-token prompt law rest on your reading of the tech report + your ASVD scan. A verification pass against the official report + HF configs either confirms or cheaply falsifies the premises before you spend 4.4 GPU-h on LoRA v2. | LoRA v2 layer set; E4B prompt cap; PLE freeze assumption |
| R6 | FastAPI/Rich/Requests test-suite forensics | **P2** | Targeted-pytest law, flaky-test knowledge (anyio, dirty_equals, expired certs, digest auth) is partly documented in those projects' issue trackers — faster than rediscovering via rollouts. | B8 test-selection prompt law; local sandbox parity |

**NOT worth spending research on (your internal evidence is already stronger than any paper):**

1. **LoRA layer selection** — your 2,560-vector ASVD + AtP-KL scan plus the Gate G2 live tournament (A/B/C/base) is *direct* empirical evidence on your exact model. No paper can beat it. Keep it.
2. **B4 vs B6 vs B7 RCA** — already root-caused with byte-level diffs and the LB ground truth (0.12/0.08/0.10). Done. Only verify on Kaggle (the B8 submission itself is the verification).
3. **Local sandbox defects** (dirty_equals, `-p no:anyio`, expired TLS mock, Py3.11 Protocol bug, POSIX zip) — already discovered and fixed in-repo. Literature adds nothing.
4. **Quota ledger / scheduling** — arithmetic, not research.

---

## 1. How to use this pack

- **Tool pairing:** R1, R3, R5 → ChatGPT Deep Research or Gemini Deep Research (literature synthesis). R2 → Gemini/Perplexity (web + forum mining). R4 → any strong model with web access; paste your exact version numbers. R6 → Perplexity/GitHub-issue search.
- **Run R1 + R2 today, in parallel.** They gate the Day 3–5 decisions (B8 authoring + LoRA v2 corpus design). R3/R4 within the next 24–36 h. R5 before the Nov 25 team-merger freeze. R6 whenever convenient before Day 7.
- Each prompt is **self-contained** — the other AI cannot see your repo, so every prompt carries the minimal project facts it needs. Paste verbatim. If you trim anything, trim the Context bullets, never the Questions.
- **When results return:** save each as a dossier under `Others/`, then make every recommendation pass your own Gate-G2-style holdout (≥ +3 solve delta, zero invariant violations) before it touches any submission. External advice is a hypothesis, not ground truth — your Gate G2 process is the arbiter.

---

## 2. The prompts (copy everything inside a block, verbatim)

### R1 — P0 · Small-model agentic SFT/LoRA science (E4B LoRA v2/v3)

**Feeds:** Day 4 LoRA v2 training (corpus 50–70 traces), Day 6 TIES/DARE merge, Day 10 final Slot 2 champion.

```
ROLE: You are a research analyst specializing in fine-tuning small language models (2–9B, incl.
LoRA/PEFT) for autonomous coding agents (SWE-bench-style tool-using agents). I need a
literature-grounded, decision-ready brief — not generic advice.

CONTEXT (my exact setup — treat as fixed constraints, do not re-litigate):
- Competition: offline agentic repo-level bug fixing on Python repos (FastAPI, Rich, Requests,
  HTTPX) inside air-gapped Docker; binary all-or-nothing scoring per task (all FAIL_TO_PASS and
  PASS_TO_PASS tests must pass; 0 fail/err/skip allowed).
- Student model: "gemma-4-e4b-it-qat-w4a16-ct" — a ~4.5B-effective / ~8B-stored edge model,
  Compressed-Tensors quantization-aware-trained W4A16 checkpoint, 42 transformer layers, 35 of
  them local-attention with a 512-token sliding window, global layers at [5,11,17,23,29,35,41],
  per-layer embeddings (PLE) tables, QAT must be dequantized to bf16 for training (NF4 adapters
  collapsed scores from 15% to 5% — empirically proven, do not revisit).
- Teacher: same family 31B model, zero-LoRA, currently solves 27/99 training tasks WITH golden
  localization hints; 50% patch precision when it emits patches on diagnostics.
- Current student result after LoRA v1: 5/30 holdout solves (base floor 1/30). LoRA v1 config:
  native QAT bf16 dequant, layers [23,29,30,35,41], all 7 linear projections, r=8, alpha=16,
  lr 7e-5, 2 epochs, chunked cross-entropy, assistant-only loss masking, corpus = 26–30 clean
  multi-turn tool-calling trajectories (~92k active assistant tokens) filtered by 8 guards
  (no test-file mutations, no holdout leakage, valid tool-call JSON, decisive submit_patch ends).
- Compute budget for v2: ONE 0.3h wall-clock training run on 4x L4 (96 GB) + ONE 0.8h 30-task
  holdout tournament. Local 16 GB GPU can generate on-policy rollouts 24/7 for free.
- Deadline: final submission Dec 2, 2026. I can realistically train ~2 more adapter versions.

QUESTIONS (answer all, in order, with citations):
1. Corpus size curve: published agent-SFT work (SWE-Gym, AgentTuning, FireAct, AgentInstruct,
   Toolformer-style, OpenHands/SWE-agent fine-tuning, distilled-coder agents, ANY 2024–2026
   paper fine-tuning ≤9B models on agentic coding trajectories): how many trajectories / active
   tokens did they use, at what model size, and what diminishing-returns threshold did they
   report? Give me a table. Then: given 26→50→70 trajectories at 4.5B-effective parameters,
   what does the literature predict for marginal solve-rate gain per +10 trajectories?
2. Trajectory composition: evidence on mixing teacher (hinted, successful) vs student
   (on-policy, error-recovery) trajectories — DAgger-style mixing ratios, upweighting of
   recovery/refinement turns, filtering only-success vs including failures. What composition
   and per-turn weighting is best supported by evidence for tool-using agents?
3. LoRA specifics on small data: rank/alpha/lr guidance for <100k supervised tokens; dropout;
   whether targeting attention (q,k,v,o) + MLP (gate,up,down) on ~5 mid-upper layers is
   supported or contradicted by published layer-importance studies for coding agents; NEFTune
   or other noise injection benefits at this data scale?
4. After SFT, is a preference/RL step (DPO/KTO/GRPO on成败 trajectories, self-improve loops)
   worth ONE additional run at my scale? Cite concrete evidence of gains for ≤9B coding agents
   with similarly tiny preference sets. If marginal, say so plainly.
5. Adapter merging: TIES / DARE / task-vector merging for LoRA adapters of the same base —
   published success/failure conditions, recommended density (k) and scaling (lambda) for
   merging a "formatting prior" adapter with a "logic" adapter, and known failure modes
   (NaNs, rank mismatch). Verdict: is merging likely to beat simply training one bigger-
   corpus adapter at my scale?
6. Small-context regime: my student has a 512-token local attention window on 35/42 layers and
   a system prompt capped at 497 tokens. Evidence on prompt distillation / ultra-short agent
   prompts (~100 tokens) — does compressing laws into fewer tokens plus LoRA outperform
   keeping a 497-token prompt with LoRA?
7. What ONE change to my v2 training run has the highest expected +solves per GPU-hour,
   and what is the single most common way teams like mine overfit at this data scale?

EVIDENCE RULES: cite papers/docs with URLs and years; distinguish {verified in source} vs
{inferred by you}; if a claim is folklore without a source, label it FOLKLORE.
NAMING NOTE: my checkpoints are internal names. Public analogs: Gemma 3n E4B/E2B (edge, PLE,
sliding-window local attention), Gemma 3 27B, Qwen2.5-Coder, Llama-3.1-8B, StarCoder2. Treat
close architectural analogs as valid evidence and state the mapping explicitly. SWE-bench /
SWE-bench-Verified / Terminal-Bench / SWE-Gym / HumanEval-Fix style benchmarks are all valid
evidence domains.
DELIVERABLE: (a) evidence table per question, (b) one final "Recommended v2 config" block
(corpus size, composition %, weighting, LoRA params, epochs, lr schedule) with confidence
level per line, (c) a ranked top-5 list of changes by expected solve delta, (d) list of
things I should explicitly NOT do at this data scale.
DO NOT: give generic "use more data" advice without the curve; propose anything needing more
than my stated compute budget; recommend NF4/bitsandbytes anything.
```

---

### R2 — P0 · Kaggle competition & leaderboard intelligence

**Feeds:** B8/B9 score targeting, Nov 25 hard-pivot circuit breaker, small-model track strategy, final 2-selection plan.

```
ROLE: You are a competitive-intelligence analyst for Kaggle ML competitions. Mine the web
(Kaggle discussion forums, public notebooks, Discord/Reddit summaries, blog write-ups, LinkedIn
posts of competitors) and produce a decision-ready intelligence brief on this competition.

CONTEXT:
- Competition: Kaggle "Google Gemma 4 Developer Agent" — $75,000 total ($65k main track any
  allowed model up to gemma-4-31b class; $10k small-model bonus restricted to the E4B/E2B
  ~4.5B edge checkpoints). Autonomous coding agents repair repo-level bugs in Python repos
  (FastAPI, Rich, Requests, HTTPX) offline inside Docker; Google ADK declarative submissions;
  vLLM serving on 4x L4. Scoring = binary resolution rate. Public LB ≈ 114 tasks.
- Current public LB: top 3 ≈ 0.29 / 0.248 / 0.242. My best (31B, zero-LoRA, generous budgets
  180s/40-calls, targeted -k pytest) = 0.12 (14/114). Small-model bonus track: rivals tied at
  ~0.10; my E4B floor = 0.05; my LoRA'd E4B = 5/30 on uncontaminated holdout (16.67%).
- Rules: 1 submission/day; exactly 2 final selected submissions; ties broken by EARLIER
  submission timestamp; team merger deadline Nov 25 2026; final deadline Dec 2 2026.

QUESTIONS:
1. Find the competition's Kaggle pages (overview/rules/discussion/leaderboard). If pages are
   gated or renamed, find the closest public Kaggle agentic-coding competitions (e.g. SWE-bench
   style, Terminal-Bench style, "Google ... agent" comps) and say so explicitly.
2. What are top teams publicly saying: discussion posts, notebooks, write-ups about their
   scores/architecture (model choice, LoRA vs not, prompt strategy, test-selection tricks,
   budget settings, concurrency)? Extract per-team tactic lists with source links.
3. What is the revealed task mix of the public split (repos, task counts), and any disclosed
   private-split hints?
4. Interpret the score economics: with binary per-task scoring and ~114 public tasks, what
   tactics plausibly move 0.12 → 0.20+ vs 0.24+? Look for write-ups quantifying prompt-only
   vs fine-tuned deltas on comparable agentic comps.
5. Small-model bonus track: any public evidence of what the 0.10-tier E4B/E2B teams are doing?
   What is the realistic ceiling people report for ~4B-class models on SWE-style benchmarks?
6. Rules edge cases: documented cases of teams losing prize eligibility via wrong final-2
   selection, dual-same-model selections, or late submissions; how winning teams time final
   submissions for the tie-break seniority rule.
7. Known infrastructure gotchas reported by competitors in discussions for this or directly
   analogous comps (evaluation timeouts, notebook exceptions, packaging, quota resets).

EVIDENCE RULES: every claim gets a URL; date-stamp each source; mark {directly about this
competition} vs {analogous competition}; if you cannot find something, say NOT FOUND — never
invent. Prefer primary sources (Kaggle itself) over secondary summaries.
DELIVERABLE: (a) per-team tactic table, (b) "what separates 0.12 from 0.24" hypothesis list
ranked by evidence strength, (c) small-model track ceiling verdict, (d) rules/gotcha checklist
for the final 10 days, (e) a one-paragraph "should I keep pushing Slot 1 or pivot quota to
Slot 2" recommendation given my numbers.
DO NOT: speculate without labels; give me motivation; repeat the rules I already stated back
to me as if they were findings.
```

---

### R3 — P1 · SWE-agent harness & prompt state-of-the-art (2024–2026)

**Feeds:** Day 3 B8 authoring, Day 7 B8b node-level pytest fallback, E4B prompt distillation.

```
ROLE: You are a research analyst on coding-agent harnesses. Produce a techniques brief for a
constrained SWE-bench-style agent: my "harness" is FIXED (I cannot change the runtime), so I
need prompt-level and policy-level techniques that transfer into a fixed 9-tool environment.

CONTEXT (my fixed environment — do not propose harness rewrites):
- 9 tools: run_command (bash in /workspace, 5,000-char output cap, timeout I control via
  config), read_file (max 150 lines / 10k chars per call; needs explicit start_line/end_line),
  edit_file (3-tier exact/flexible/regex string replacement; fails with NoMatchError on
  whitespace drift, copied line numbers, stale post-edit context), write_file, get_status
  (free), submit_patch (free, ends session), get_code_neighbors / search_similar_code /
  get_code_subgraph (AST graph + symbol embeddings; query must be a dotted symbol path).
- Agent: single LlmAgent (root-composite agents crash — banned), declarative YAML submission,
  model is a QAT INT4 Gemma-class model served via vLLM with a gemma-style tool-call parser;
  conversation history compacts at 14,336 tokens (I keep thoughts out of history — mandatory).
- Per-task budgets I control: command timeout (180 s), tool calls (40), wall clock (6 min),
  thinking budget (768–1024), output cap (4096).
- My failure taxonomy from 147 mined traces: NoMatchError = 92.3% of tool errors; right-file-
  logic-fail 37.2%; no-edit-empty 30.8%; wrong-file localization 24.4%; 29% of turns sometimes
  emit planning prose instead of tool calls; models submit after py_compile without pytest;
  models chase wrong hypotheses instead of mirroring adjacent-function idioms.

QUESTIONS:
1. Localization policy: published localization pipelines (Agentless 3-stage, SWE-agent search
   policies, Moatless tree search, AutoCodeRover AST-guided, OpenHands) — what file/target
   identification accuracy do they reach, in how many steps, and which techniques transfer
   into my 9-tool contract without new tools?
2. Test-selection policy: evidence-based techniques for choosing WHICH test to run and how
   to filter (node-level pytest selection, -k filters, using the issue text to guess test
   names, avoiding full-file sweeps) — measured time savings and risk of missing
   PASS_TO_PASS regressions. What do top harnesses do about pre-existing failing tests?
3. Edit reliability: known prompt/behavioral mitigations for exact-match string editing
   failures (line-number pollution, whitespace drift, stale buffers) — e.g., re-read-before-
   edit, small-hunk policy, anchor lines, copy-exact-whitespace rules. Any measured success?
4. Context discipline: techniques to survive ~14k-token compaction and a 512-token local
   attention window (my small-model case): progressive summarization, scratchpad-in-/tmp,
   state re-statement patterns, "notebook" conventions agents keep in the prompt. Evidence
   for ultra-short system prompts (<150 tokens) in agentic coding?
5. Anti-pathologies: published fixes for (a) no-tool-call turns / planning prose, (b) premature
   submit after syntax check only, (c) hypothesis-chasing vs adjacent-mechanism mirroring,
   (d) scratch-file pollution of git diff. Which of these have measured results vs folklore?
6. Pacing: evidence on first-edit timing vs solve rate in agentic SWE (my data says first
   edit by step 8–10 → 100% patch emission, 67% of solves). Does published work corroborate
   or contradict an "early educated edit" policy?
7. Give me the 8 laws of a distilled ~100-token system prompt for the small model, and a
   ~3,000-char prompt skeleton for the 31B model, each law tagged with its evidence source
   and confidence.

EVIDENCE RULES: cite URLs + years; tag each technique {measured in paper} vs {practitioner
folklore} vs {your inference}. Include SWE-bench-Verified top-solution write-ups (2024–2026),
SWE-agent/Agentless/OpenHands/Moatless/AutoCodeRover docs and papers, Terminal-Bench reports.
DELIVERABLE: (a) technique table with transferability verdict for MY toolset, (b) the two
prompt skeletons, (c) top-5 highest-confidence changes I should make to my current prompt
laws, (d) anything in current SOTA practice I am NOT doing that is cheap and low-risk.
DO NOT: propose new tools, agent frameworks, retrieval systems, or anything requiring code
changes outside my prompt/config; restate my own constraints as recommendations.
```

---

### R4 — P1 · vLLM + Compressed-Tensors QAT + PEFT serving stack

**Feeds:** Master Engine Phase 3/4 correctness, any 31B serving change, torchao/version pinning, final packaging.

```
ROLE: You are an inference-stack engineer researching vLLM + quantized checkpoints + LoRA
serving. Produce a version-locked compatibility and tuning brief.

CONTEXT (my exact stack):
- Serving: vLLM on Kaggle 4x L4 (96 GB total, Ada sm_89), tensor_parallel_size=4,
  gpu_memory_utilization=0.90, max_model_len=32768, tool_call_parser="gemma4"-style,
  reasoning_parser="gemma4"-style, OpenAI-compatible server, LiteLLM with drop_params=True.
- Checkpoints: Compressed-Tensors quantization-aware-trained W4A16 (INT4 weights, bf16
  activations) checkpoints of a Gemma-class 31B and a ~4.5B-effective edge model with
  per-layer embeddings and a small multi-token-prediction speculative drafter.
- LoRA: PEFT adapters trained after dequantizing the QAT checkpoint to bf16 (native QAT
  dequant — bitsandbytes NF4 adapters on these checkpoints collapsed accuracy; empirical,
  do not revisit). Adapters served via vLLM enable_lora with max_lora_rank<=128.
- Already hit one landmine: PEFT's torchao availability check rejected torchao 0.10.0
  (needs >0.16.0) mid-run; we now uninstall torchao in the notebook before training.
- One prior crash signature: "ImportError: Found an incompatible version of torchao ... only
  versions above 0.16.0 are supported" from PEFT is_torchao_available().

QUESTIONS:
1. Compatibility matrix (with versions and dates): vLLM support for compressed-tensors
   quantized checkpoints + LoRA composition — which vLLM versions support LoRA on top of
   W4A16 compressed-tensors models, with which quant formats (int4? fp8? pack-quant?),
   and what are the open GitHub issues about LoRA + compressed-tensors weight mismatch or
   activation drift?
2. Known issues specifically for tensor_parallel_size=4 + LoRA on L4-class GPUs (rank
   splitting, sleep/wake, CUDA graph capture with LoRA, multi-LoRA dynamic scheduling) —
   what breaks and recommended workarounds/flags?
3. The torchao/PEFT version pin: which torchao versions satisfy PEFT's check, and is
   uninstalling torchao the right call or does a pin >=0.16.0 buy anything (e.g., torchao
   LoRA paths) for bf16 dequantized training?
4. Speculative decoding: for a model with a built-in MTP-style drafter, what are the
   correct vLLM flags on TP=4 (speculative_config / draft model usage), known bugs with
   LoRA + speculative decoding, and measured speedups on L4-class hardware?
5. Throughput tuning for my shape: 4 concurrent agent sessions, multi-turn, long prompts,
   ~4k max output, thinking tokens parsed by a reasoning parser. Best-practice settings:
   KV-cache dtype on L4, enable_prefix_caching (any known conflicts with LoRA or sliding-
   window local attention?), max_num_seqs, chunked prefill — cite issues/docs.
6. Serving the 31B QAT W4A16 checkpoint on 4x L4: reported VRAM footprints and pitfalls
   (weights not evenly sharding, per-GPU ~24GB headroom, activation spikes at 32k context,
   OOM signatures and mitigations).
7. Packaging: known vLLM/PEFT adapter-format incompatibilities when shipping adapters inside
   a zip (adapter_config.json fields vLLM's LoRA loader actually reads — target_modules as
   list vs regex, layers_to_transform handling) and failure signatures.

EVIDENCE RULES: cite GitHub issues (with issue numbers), docs pages, release notes with
URLs and dates; state the exact version each claim applies to; if behavior differs across
versions, give the version-conditional answer; mark NOT FOUND where applicable. Public
analog naming is fine: Gemma 3 / Gemma 3n QAT W4A16 checkpoints on vLLM, Llama QAT, AWQ,
GPTQ — the compressed-tensors + LoRA question is the same class of problem.
DELIVERABLE: (a) version-pinned compatibility matrix, (b) recommended pinned stack
(vllm/peft/transformers/torchao) for Kaggle notebooks as of today, (c) flag checklist for
my exact serving shape, (d) top 5 known bugs I should defensively code around.
DO NOT: recommend switching serving engines (TGI/llama.cpp are not available to me);
recommend un-quantizing to fp16 for the 31B (does not fit); give undated advice.
```

---

### R5 — P2 · Gemma architecture ground-truth verification

**Feeds:** LoRA v2 layer set, E4B prompt cap, PLE freeze, KV/attention assumptions in the ML research doc.

```
ROLE: You are a model-architecture fact-checker. Verify a short list of specific architectural
claims against primary sources (the Google technical report / model card / HF config / vLLM
implementation). Mark each claim VERIFIED / CORRECTED / NOT FOUND with the exact source quote
and URL.

CONTEXT: I am fine-tuning a Gemma-class edge model ("E4B", ~4.5B effective / ~8B stored, QAT
W4A16 INT4) and a 31B sibling. My internal docs make the following claims that drive real
money (LoRA layer targeting and prompt sizing), so I need them verified or corrected:

CLAIMS TO VERIFY (one verdict each):
1. The E4B model has 42 transformer layers, of which 35 use local sliding-window attention
   with a 512-token window, and 7 global-attention layers at indices [5, 11, 17, 23, 29, 35, 41].
2. Global layers use proportional RoPE (p-RoPE, p≈0.25) with an extended base frequency;
   local layers use standard RoPE base 10,000.
3. Layers ~24–41 share KV cache projections; keys and values remain distinct (no "values =
   keys" optimization).
4. Per-Layer Embeddings (PLE): ~670M params in per-layer embedding tables plus a large main
   embedding; PLE and the frozen multimodal encoders (ViT, USM) should stay frozen during
   text-only LoRA — any published fine-tuning guidance confirming PLE instability when trained?
5. A built-in multi-token-prediction speculative drafter exists (~4 layers, dim 256).
6. The 31B model has 64 layers with hybrid sliding (1024/2048-token) local windows and can
   tolerate system prompts of 4–6k characters without instruction-drift.
7. QAT W4A16 checkpoints are quantization-aware-trained with linear integer quantization,
   dynamic block-wise scaling and zero points — and adapters trained on bitsandbytes NF4
   then mounted on these checkpoints cause measurable degradation (any public reproductions
   of this NF4-vs-QAT mismatch beyond my own?).
8. Sliding-window local attention interacts with vLLM prefix caching — any known correctness
   or performance issues serving 512-token-window models on vLLM?

QUESTIONS beyond verdicts:
9. If any layer indices / window sizes / counts differ from the claims above, give the
   corrected numbers and what that changes for LoRA targeting of "upper global layers".
10. Published mechanistic-interpretability evidence on which layers encode tool-calling /
    instruction-following behavior in mid-size Gemma-class models (anything on layers ~23–41)?

EVIDENCE RULES: primary sources only (Google tech report arXiv page, model cards on HF,
config.json fields, vLLM/transformers implementation files, Google blog posts); quote the
exact line/field; give URLs; date each source; if the exact "Gemma 4" naming is not public,
verify against the closest public analog (Gemma 3n E4B/E2B technical report, Gemma 3 report)
and STATE the mapping. NOT FOUND is an acceptable verdict — do not guess.
DELIVERABLE: verdict table (claim | verdict | source quote | URL | what it changes for me),
then a 5-line "net impact on my LoRA v2 config" summary.
DO NOT: paraphrase from memory without a URL; blend generations of models together.
```

---

### R6 — P2 · FastAPI / Rich / Requests / HTTPX test-suite forensics

**Feeds:** B8 targeted-pytest law, local sandbox parity, regrade harness fidelity.

```
ROLE: You are a test-suite forensics analyst. I run an agent that must run targeted tests
inside specific Python repos at fixed commits, offline, on a 2-vCPU container. I need a
practical map of each repo's test suite: what runs fast, what is flaky/environment-broken,
and what the minimal reliable test-selection commands are.

CONTEXT: repos and versions are mature Python libraries: FastAPI (with Starlette/anyio/pytest
fixtures, dirty-equals based assertions in some tests), Rich, Requests, HTTPX. My agent has
bash + pytest only, ~180s per command, and MUST run only the tests relevant to a bug fix
(node-level selection) without tripping unrelated failures.

QUESTIONS:
1. FastAPI: (a) why does `-p no:anyio` break async test collection (exact mechanism, issue
   threads)? (b) which tests require dirty-equals / inline-snapshot and what happens at
   collection time when missing? (c) known slow or environment-dependent tests (network,
   SSL certs, digest auth) at recent versions; (d) minimal commands for node-level selection
   and their typical runtimes.
2. Requests: (a) the known flaky/broken tests around the built-in mock httpbin handler,
   digest auth, and EXPIRED TLS certificates (publicly reported?) — exact test names and
   issue threads; (b) how to select a single failing test node and avoid the full
   test_requests.py sweep; typical collection vs execution time split.
3. Rich: known environment-sensitive tests (terminal width, platform-specific rendering,
   encoding) and the safest node-level invocation patterns.
4. HTTPX: tests known to require network vs offline-safe subsets; sync vs async client test
   selection; common fixture requirements.
5. Cross-cutting: (a) known pytest plugin landmines when running single nodes (-p no:anyio,
   -p no:randomly, anyio fixture modes, asyncio_mode settings); (b) whether `--junitxml`
   changes test behavior in any of these suites; (c) safest "collect-only first, then run
   selected nodes" two-step pattern an agent can use.
6. For each repo: a table of the 10 most environment-fragile test files/tests with the
   environment requirement and a workaround (skip filters, -k expressions, deselects) that
   keeps the run honest (no hiding real regressions).

EVIDENCE RULES: cite GitHub issues/PRs/test files with URLs; quote test names exactly as
they appear; if a claim applies only to certain versions, state the version range; mark
NOT FOUND where applicable.
DELIVERABLE: per-repo cheat-sheet my agent can encode into prompt laws: fast/reliable
node-selection commands, known-broken tests to deselect explicitly, environment
requirements, and the two-step collect-then-run pattern.
DO NOT: recommend installing anything at runtime (air-gapped); recommend modifying test
files; give generic pytest tutorials.
```

---

## 3. Plug-back plan — where each research output lands

| Research | Output lands in | Decision it changes | Deadline to have it |
|:--|:--|:--|:--|
| R1 | `Others/` dossier + Day 4 training config | LoRA v2 corpus size/composition, rank/alpha, merge-or-not, DPO-or-not | **Nov 24** (before Day 4 training run) |
| R2 | `Others/` dossier + B8 target recalibration | B8/B9 targeting; Nov 25 pivot decision; final-2 selection plan | **Nov 24** (before Nov 25 circuit breaker) |
| R3 | Prompt law edits for B8 + E4B prompt distillation | Day 3 B8 authoring; 497→~100-token prompt attempt | Nov 25–26 |
| R4 | Version pins + serving flag checklist in Master Engine | Phase 3/4 robustness; torchao handling | Nov 26 |
| R5 | Verdict table appended to ML research doc | LoRA v2 layer set (confirm [23,29,30,35,41] or adjust); E4B prompt cap | Nov 26–27 |
| R6 | Test-selection cheat-sheet into B8 prompt + local sandbox parity list | Day 7 node-level fallback calibration | Nov 28 |

**Golden rule when integrating any result:** external research output enters the repo as a *hypothesis with citations*, and every change to a submission must still clear your existing promotion gates (local holdout ≥ prior champion, Gate-G2-style ≥ +3 delta for Slot 2, POSIX/ADK compliance checks). Nothing goes straight from a research report to a Kaggle submission without passing your own holdout first — the B5 "surgical laws" and the external AI "29.8h plan" you already triaged are exactly why.

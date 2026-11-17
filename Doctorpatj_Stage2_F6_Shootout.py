# %% [markdown]
# # Kaggle Gemma 4 Developer Agent — Stage 1 Controlled `31B` 2-Rung A/B Shootout
# **Target Model:** `gemma-4-31b-it-qat-w4a16-ct` ($65k Main Track — Zero Weight Tuning)
# **Structure:** Strictly 6 Clean Self-Contained Cells (Zero trailing overwrite cells):
# - **Cell 1:** Offline Wheelhouse, `inline_snapshot`, and Live Localhost `httpbin` Shims (Run once)
# - **Cell 2:** Build Controlled `31B` Staircase (`C1_B0_budget40`, `C1a_31B_tb1024`, `C1b_31B_tb1024_direct`)
# - **Cell 3:** Start vLLM Server on 4x L4 GPUs (`tp=4`, `bfloat16`, `max_model_len=32768`)
# - **Cell 4:** Controlled `DIAGNOSTIC_20` A/B Shootout at `CONCURRENCY = 2` (`C1a` vs `C1b`)
# - **Cell 5:** Upgraded Doctor Table + Auto-Export `/kaggle/working/doctor_winner_dump.txt` (`< 1.5 MB`)
# - **Cell 6:** Package & Strictly Validate Shootout Winner (`CHAMPION_NAME`) into `/kaggle/working/submission.zip`

# %% [markdown]
# ## Cell 1: Offline Environment Setup, Wheelhouse Install & Local Sandbox Shims

# %%
import glob
import importlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

# 1. Configure offline vLLM serving and LiteLLM routing environment variables
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENGINE_READY_TIMEOUT_S"] = "1200"
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["LITELLM_LOG"] = "WARNING"
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "EMPTY")
os.environ["LOCAL_API_KEY"] = os.environ.get("LOCAL_API_KEY", "EMPTY")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_PYTHON_DISABLED"] = "true"


def configure_logging() -> None:
    """Silence verbose third-party loggers."""
    for name in (
        "LiteLLM",
        "LiteLLM Router",
        "LiteLLM Proxy",
        "litellm",
        "httpx",
        "httpcore",
        "openai",
        "urllib3",
        "opentelemetry",
        "opentelemetry.context",
    ):
        lg = logging.getLogger(name)
        lg.setLevel(logging.WARNING)
        if "litellm" in name.lower():
            lg.propagate = False
            lg.handlers.clear()


configure_logging()

DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
WHEELHOUSE_DIR = Path("/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse")
DIST_PKG = Path("/usr/local/lib/python3.12/dist-packages")

# 2. Remove broken cutlass .pth hooks if present
for pth_pattern in (
    "/usr/local/lib/python*/dist-packages/*cutlass*.pth",
    "/usr/local/lib/python*/site-packages/*cutlass*.pth",
):
    for pth in glob.glob(pth_pattern):
        try:
            os.unlink(pth)
        except OSError:
            pass

# 3. Restore PEP 440 '+cu128' wheel filenames and install main wheelhouse
tmp_whl = Path("/tmp/wheelhouse")
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob("*.whl"):
    if "cutlass" in w.name.lower():
        continue
    target_name = (
        w.name.replace("cu128", "+cu128")
        if ("cu128" in w.name and "+" not in w.name)
        else w.name
    )
    target = tmp_whl / target_name
    if not target.exists():
        try:
            os.symlink(w, target)
        except OSError:
            shutil.copyfile(w, target)

wheels = sorted(str(w) for w in tmp_whl.glob("*.whl"))
print(f"Installing {len(wheels)} wheels from {WHEELHOUSE_DIR}...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels],
    check=True,
)

# 4. Install extra competition test helper wheels (deduplicated by latest version; never touch core pytest!)
from packaging.version import Version

best_helper_wheels = {}
for w in glob.glob(str(DATA_DIR / "wheels" / "*.whl")):
    fname = Path(w).name
    parts = fname.split("-")
    if len(parts) < 2:
        continue
    pkg_norm = parts[0].lower().replace("_", "-")
    if pkg_norm in ("pytest", "pluggy", "iniconfig", "packaging"):
        continue
    if any(k in pkg_norm for k in ("snapshot", "dirty", "httpbin", "trustme", "werkzeug")):
        try:
            ver = Version(parts[1])
        except Exception:
            continue
        if "cp3" in fname and "cp312" not in fname and "abi3" not in fname:
            continue
        if pkg_norm not in best_helper_wheels or ver > best_helper_wheels[pkg_norm][0]:
            best_helper_wheels[pkg_norm] = (ver, w)

for pkg_norm, (_, whl_path) in sorted(best_helper_wheels.items()):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", whl_path], check=False)

# 5. Install inline_snapshot fallback shim if not already installed
try:
    import inline_snapshot  # noqa: F401
except ImportError:
    (DIST_PKG / "inline_snapshot.py").write_text(
        "def snapshot(val=None):\n    return val\n",
        encoding="utf-8",
    )

# 6. Install live localhost ThreadingHTTPServer httpbin plugin ONLY if real pytest_httpbin is absent
try:
    import pytest_httpbin  # noqa: F401
    has_real_pytest_httpbin = True
except ImportError:
    has_real_pytest_httpbin = False

live_local_httpbin = """import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import pytest


class _MiniHttpbinHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _handle_all(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        content_len = int(self.headers.get("Content-Length") or 0)
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b""

        if path == "/redirect-to":
            target = qs.get("url", ["/get"])[0]
            code = int(qs.get("status_code", [302])[0])
            self.send_response(code)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path.startswith("/redirect/") or path.startswith("/relative-redirect/"):
            n = int(path.rstrip("/").split("/")[-1])
            loc = "/get" if n <= 1 else f"/relative-redirect/{n - 1}"
            self.send_response(302)
            self.send_header("Location", loc)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path.startswith("/status/"):
            code = int(path.rstrip("/").split("/")[-1])
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        payload = json.dumps({
            "url": f"http://{self.headers.get('Host', '127.0.0.1')}{self.path}",
            "args": {k: v[0] if len(v) == 1 else v for k, v in qs.items()},
            "headers": dict(self.headers),
            "data": body_bytes.decode("utf-8", errors="replace"),
            "origin": "127.0.0.1",
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    do_GET = _handle_all
    do_POST = _handle_all
    do_PUT = _handle_all
    do_DELETE = _handle_all
    do_PATCH = _handle_all
    do_HEAD = _handle_all
    do_OPTIONS = _handle_all


class _LocalHttpbinURL:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"

    def __str__(self):
        return self.url

    def __add__(self, other: str):
        return self.url + other

    def join(self, *parts):
        suffix = "/".join(str(p).lstrip("/") for p in parts)
        return f"{self.url}/{suffix}" if suffix else self.url


@pytest.fixture(scope="session")
def httpbin():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MiniHttpbinHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    host, port = srv.server_address
    yield _LocalHttpbinURL(host, port)
    srv.shutdown()


@pytest.fixture
def httpbin_secure():
    pytest.skip("TLS httpbin_secure skipped in offline sandbox")
"""

if not has_real_pytest_httpbin:
    (DIST_PKG / "_swegemma_httpbin_shim.py").write_text(live_local_httpbin, encoding="utf-8")
    dist_info = DIST_PKG / "swegemma_httpbin_shim-0.1.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: swegemma-httpbin-shim\nVersion: 0.1\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[pytest11]\nswegemma_httpbin = _swegemma_httpbin_shim\n",
        encoding="utf-8",
    )

importlib.invalidate_caches()
configure_logging()
print("[✓] Environment, wheelhouse, inline_snapshot shim, and live localhost httpbin server ready.")

# %% [markdown]
# ## Cell 2: Controlled 2-Rung `31B` Staircase (`C1a_31B_tb1024` vs `C1b_31B_tb1024_direct`)
# Isolates **strictly one variable per rung** against Monday's `B3` anchor (`C1_B0_budget40`):
# - **`C1_B0_budget40` (Anchor)**: `40` calls, `6m`, `thinking_budget: 2048`, `code_analyzer` enabled (`4/20` at `CONCURRENCY=4`)
# - **`C1a_31B_tb1024` (Rung 1)**: Exact `C1_B0_budget40` + **ONLY** `thinking_budget: 2048 -> 1024`
# - **`C1b_31B_tb1024_direct` (Rung 2)**: Exact `C1a_31B_tb1024` + **ONLY** remove `code_analyzer` `AgentTool` (root `31B` uses all 9 tools directly)

# %%
import re
import shutil
from pathlib import Path
from adk_submission import validate_directory
from adk_submission.discovery import discover_declared_models
from swegemma.config import build_submission_limits
from swegemma.models import load_tasks

DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
WORKING_DIR = Path("/kaggle/working")
CANDIDATES_ROOT = WORKING_DIR / "candidates"
AGENT_DIR = WORKING_DIR / "sample_submission"
GRAPH_DIR = str(DATA_DIR / "graphs")
EMBEDDINGS_DIR = str(DATA_DIR / "embeddings")
TASKS_PATH = DATA_DIR / "tasks.jsonl"
MODEL_NAME = "gemma-4-31b-it-qat-w4a16-ct"

B0_SAMPLING = """temperature: 0.1
top_p: 0.90
max_output_tokens: 8192
thinking_config:
  thinking_budget: 2048
  include_thoughts: false
"""

SAMPLING_TB1024 = """temperature: 0.1
top_p: 0.90
max_output_tokens: 8192
thinking_config:
  thinking_budget: 1024
  include_thoughts: false
"""

B0_EVAL_40 = """evaluation:
  timeout_seconds: 180
  max_tool_calls: 40
  max_time_minutes: 6
  max_turns: 55
"""

B0_ANALYZER_EXACT = (
    "You are a specialized code analysis sub-agent. Your role is to inspect repository source files, "
    "trace symbol relationships, and identify the root cause of the reported issue.\n\n\n\n"
    "## Instructions\n\n"
    "1. Use `search_similar_code`, `get_code_neighbors`, `get_code_subgraph`, and `read_file` to locate "
    "the exact functions, classes, and line ranges involved in the bug.\n\n"
    "2. Provide a concise, structured report containing:\n\n"
    "   - Exact file paths and line numbers to modify.\n\n"
    "   - Root cause explanation.\n\n"
    "   - Recommended minimal code change.\n"
)

B0_SYSTEM_EXACT = (
    "You are an expert autonomous software engineer assigned to resolve an issue in a repository "
    "efficiently and decisively.\n\n\n\n"
    "## Core Objective: Fast, Minimal, and Precise Fixes\n\n"
    "Aim to understand, resolve, and submit the fix in the minimum number of tool calls (under 8–10 turns). "
    "Move directly from the problem statement to the relevant files, apply the solution, verify with a "
    "targeted test, and submit.\n\n\n\n"
    "## Workflow\n\n\n\n"
    "### 1. Identify Target Files Immediately\n\n"
    "- Extract filenames, functions, classes, CLI subcommands, or error messages directly from the problem statement.\n\n"
    "- Read only the specific target files and lines using `read_file` or search tools.\n\n\n\n"
    "### 2. Implement the Solution Directly\n\n"
    "- Apply the minimal necessary fix or feature directly to the source files using `edit_file` or `write_file`.\n\n"
    "- Strictly adhere to specified error strings, exception types, HTTP status codes, and API signatures.\n\n"
    "- For documentation code tasks (e.g. FastAPI), edit executable code under `docs_src/`.\n\n"
    "- Mirror Existing File & Test Conventions: Read adjacent functions in the target source file and its "
    "corresponding test file (tests/test_<module>.py) before editing so your exception types, error message "
    "strings, and docstrings match the repository's exact pattern\n\n"
    "- Check Hidden CI/Scripts for Automation Tasks: Since .github/ is omitted from the initial workspace tree, "
    "if a task involves release preparation, CLI scripts, or CI workflows, inspect .github/workflows/ and "
    "scripts/ using run_command (ls .github/workflows scripts).\n\n"
    "- Keep Scratch Scripts in /tmp: Because submit_patch runs git add -N . inside /workspace, never create "
    "temporary reproduction files inside /workspace; write any scratch test script to /tmp/repro.py.\n\n\n\n"
    "### 3. Run Targeted Tests Only (Existing Tests May Be Broken)\n\n"
    "- **Run ONLY Targeted Tests**: Run only the specific test file or test method directly verifying the bug "
    "or feature you modified (e.g. `pytest tests/test_target.py -k test_feature` or "
    "`python3 -m unittest tests.test_target`).\n\n"
    "- **Be Aware That Existing Tests May Be Broken**: Many repositories contain pre-existing test breakages, "
    "missing test data fixtures (e.g. `/test_data`), or environment import errors unrelated to your task.\n\n"
    "- **Do NOT Attempt to Fix Existing Tests**: If an existing test fails due to pre-existing repository issues "
    "or missing fixtures, IGNORE IT. Never spend turns attempting to repair pre-existing test failures, create "
    "test stubs, or alter test code.\n\n"
    "- **STRICT RULE: NEVER Run Bare Pytest or Full-Repo Sweeps**: NEVER run bare `pytest`, `pytest .`, "
    "`python3 -m unittest discover`, or full-repo test suites without specifying a target file. Full test suites "
    "take several minutes, cause catastrophic timeouts, and exhaust your turn and time budgets.\n\n"
    "- If you need to locate the test file, find it explicitly with `find tests -name \"*<name>*.py\"` instead "
    "of running the test runner across the repo.\n\n\n\n"
    "### 4. Immediate Patch Submission\n\n"
    "- Once your targeted test passes:\n\n"
    "  1. Call `submit_patch` immediately.\n\n"
    "  2. Verify `patch_size > 0` and `files_changed > 0`.\n\n"
    "  3. Output a short summary of the fix to end the session.\n\n\n\n"
    "## Anti-Patterns to Avoid\n\n"
    "- **NEVER modify, create, or delete test files** (`*_test.py`, `test_*.py`, or anything under `tests/`). "
    "All changes must be to source implementation files. Modifying tests results in an automatic evaluation failure.\n\n"
    "- **NEVER run full repository test suites** (e.g., bare `pytest` or `pytest .`) — always specify the exact "
    "test file path.\n\n"
    "- **NEVER attempt to fix or repair existing tests or pre-existing repository breakages** — your task is "
    "strictly to implement the fix for the reported issue in source code.\n\n"
    "- **NEVER search outside `/workspace`** for source files or packages (e.g., `/usr/local/lib/`, `/wheels/`, "
    "`/opt/`). All repository code and test dependencies are pre-installed. If `ModuleNotFoundError` occurs "
    "during test runs, focus on fixing code under `/workspace`, not looking for missing system packages.\n\n"
    "- Do NOT spend turns running broad exploratory searches if the file path or symbol is obvious.\n\n"
    "- Do NOT refactor or reformat unrelated functions or files.\n\n"
    "- Do NOT conclude without submitting a non-empty patch (`patch_size > 0`). Every task requires concrete "
    "source modifications. Concluding that the codebase is already clean without making changes is an anti-pattern.\n"
)



F6_APEX_V3_SYSTEM_MD = (
    "You are an elite autonomous Python software engineer resolving a repository issue in /workspace "
    "within a 40-call, 6-minute budget.\n\n"
    "## Core 5-Step Execution Loop\n\n"
    "1. **TURN 1 SEMANTIC + GRAPH LOCALIZATION**: Start Turn 1 with `search_similar_code` on the primary symbol, "
    "class, CLI subcommand, or error phrase from the problem statement (note the exact filepaths and line numbers "
    "returned!), then call `get_code_neighbors` to trace callers, re-exports, and internal usages across the package. "
    "For documentation example tasks, inspect executable files under `docs_src/`. For CI/release tasks, check "
    "`.github/workflows/` and `scripts/` via `run_command`.\n\n"
    "2. **WINDOWED READ_FILE + SIBLING CONVENTION GREP (CRITICAL)**: Calling `read_file` without `start_line` ONLY "
    "returns lines 1 to 150! Whenever a file exceeds 150 lines or `read_file` truncates output, ALWAYS pass explicit "
    "integer `start_line` and `end_line` (for example `start_line=151, end_line=300`) so you copy `old_string` "
    "verbatim from lines you actually read. Before adding errors, warnings, or deprecations, run `grep -rn` via "
    "`run_command` on sibling source files to mirror the repository's exact exception type, wording, or decorator.\n\n"
    "3. **4 ROOT-CAUSE ENGINEERING LAWS**:\n"
    "   - **Lowest Shared Bottleneck & Non-Breaking Deprecations**: Patch the lowest shared callee reached by all "
    "public entry points; never place imports above `from __future__ import annotations`. CRITICAL PYTHON LAW: When "
    "deprecating a class, function, or parameter, NEVER `raise DeprecationWarning` as an exception and NEVER delete "
    "or replace the working method body! Use `@deprecated` from `typing_extensions` or "
    "`warnings.warn(..., DeprecationWarning, stacklevel=2)` matching sibling files while preserving 100% of existing "
    "return behavior, and update internal package references so unrelated code does not trigger warnings under `-Werror`.\n"
    "   - **First Decision Gate**: Patch the earliest check that decides or short-circuits behavior rather than "
    "downstream helpers, preserving existing port, prefix, leading-dot, or option branches.\n"
    "   - **Null-Safe Owner State**: Gate side effects via null-safe checks on the state-owning class; never index "
    "optional tuples or attributes directly when `None` is possible.\n"
    "   - **Bare-Symbol & Twin-Package Sweep**: After changing or deprecating any symbol or method signature, grep "
    "both `name(` and bare `name` across the package and any sync/async twin modules.\n\n"
    "4. **ANTI-LOOP SURGICAL EDIT + COMPILE**: Copy `old_string` verbatim from `read_file`. If `edit_file` ever fails "
    "to match, NEVER guess `old_string` and NEVER call `read_file` without `start_line`—immediately call `read_file` "
    "with explicit `start_line` and `end_line` on that exact 60-line window (or `grep -n` via `run_command`). Run "
    "`python3 -m py_compile` on edited source files to catch any escaping or syntax errors immediately.\n\n"
    "5. **INLINE FALSIFICATION + MANDATORY TARGETED PYTEST BEFORE SUBMIT**:\n"
    "   - **NEVER write files to `/tmp` (such as `/tmp/repro.py`) or leave untracked scratch scripts in `/workspace`** "
    "(container path isolation blocks `/tmp` or pollutes `git diff`). Always run inline checks via `run_command` with "
    "`python3 -c \"...\"` testing BOTH the reported bug/edge case AND valid return values.\n"
    "   - **HARD RULE — NEVER CALL `submit_patch` ON `py_compile` ALONE**: First locate the exact test file covering "
    "your modified symbol (`grep -rn \"SymbolName\" tests/` or `find tests -name \"*mod*.py\"`), then run ONLY that "
    "targeted test file (e.g. `pytest tests/test_target.py -k test_case` — NEVER run bare `pytest` across the whole repo!).\n"
    "   - **Ignore Unrelated Pre-Existing Test Breakages**: NEVER modify, create, or delete files under `tests/`, and "
    "NEVER run `git checkout` to revert a valid source fix because of unrelated mock/network/fixture test failures. "
    "Once your inline check and targeted bug test pass and `git diff` is clean and non-empty, call `submit_patch` immediately.\n"
)


def _make_31b_variant(
    name: str,
    eval_yaml: str,
    sampling_yaml: str,
    sys_md: str,
    use_analyzer: bool,
) -> Path:
    cdir = CANDIDATES_ROOT / name
    if cdir.exists():
        shutil.rmtree(cdir)
    subs = ("configs", "prompts", "sub_agents") if use_analyzer else ("configs", "prompts")
    for sub in subs:
        (cdir / sub).mkdir(parents=True, exist_ok=True)

    (cdir / "eval_config.yaml").write_text(eval_yaml, encoding="utf-8")
    (cdir / "configs" / "sampling.yaml").write_text(sampling_yaml, encoding="utf-8")
    (cdir / "prompts" / "system.md").write_text(sys_md, encoding="utf-8")

    analyzer_tool_block = ""
    if use_analyzer:
        (cdir / "prompts" / "analyzer.md").write_text(B0_ANALYZER_EXACT, encoding="utf-8")
        (cdir / "sub_agents" / "code_analyzer.yaml").write_text(
            f"""name: code_analyzer_agent
description: Analyzes repository source files and symbol graphs to locate root causes.
model: {MODEL_NAME}
instruction: !include ../prompts/analyzer.md
tools:
  - read_file
  - search_similar_code
  - get_code_neighbors
  - get_code_subgraph
generate_content_config: !include ../configs/sampling.yaml
""",
            encoding="utf-8",
        )
        analyzer_tool_block = """  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true
"""

    (cdir / "agent.yaml").write_text(
        f"""name: swe_gemma4_{name.lower()}
model: {MODEL_NAME}
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
{analyzer_tool_block}generate_content_config: !include configs/sampling.yaml
""",
        encoding="utf-8",
    )

    limits, _ = build_submission_limits()
    validate_directory(cdir, limits=limits)
    declared = discover_declared_models(cdir)
    assert declared == {MODEL_NAME}, f"Model mismatch in {name}: {declared}"
    for md in (cdir / "prompts").glob("*.md"):
        assert not re.findall(r"\{([^{}]+)\}", md.read_text(encoding="utf-8")), f"Illegal brace in {md.name}"
    print(f"  [✓] Built & validated: {name} (analyzer={use_analyzer})")
    return cdir


if CANDIDATES_ROOT.exists():
    shutil.rmtree(CANDIDATES_ROOT)

CANDIDATE_DIRS = {
    "C1_B0_budget40": _make_31b_variant(
        "C1_B0_budget40", B0_EVAL_40, B0_SAMPLING, B0_SYSTEM_EXACT, use_analyzer=True
    ),
    "C1a_31B_tb1024": _make_31b_variant(
        "C1a_31B_tb1024", B0_EVAL_40, SAMPLING_TB1024, B0_SYSTEM_EXACT, use_analyzer=True
    ),
    "C1b_31B_tb1024_direct": _make_31b_variant(
        "C1b_31B_tb1024_direct", B0_EVAL_40, SAMPLING_TB1024, B0_SYSTEM_EXACT, use_analyzer=False
    ),
    "C2_31B_F6_Apex_v3": _make_31b_variant(
        "C2_31B_F6_Apex_v3", B0_EVAL_40, SAMPLING_TB1024, F6_APEX_V3_SYSTEM_MD, use_analyzer=False
    ),
}

init_stage = "C2_31B_F6_Apex_v3"
if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)
shutil.copytree(CANDIDATE_DIRS[init_stage], AGENT_DIR)
CHAMPION_NAME = init_stage

tasks = load_tasks(TASKS_PATH)
task_map = {t.instance_id: t for t in tasks}
print(f"[✓] Ready: {list(CANDIDATE_DIRS.keys())} ({len(tasks)} tasks loaded, default={CHAMPION_NAME})!")

# %% [markdown]
# ## Cell 3: Start vLLM Server on 4x L4 GPUs (`tp=4`, Zero LoRA Overhead)

# %%
import re
import litellm
import torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

litellm.drop_params = True

ALLOWED_MODEL_NAMES = frozenset({
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
})
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
INFERENCE_API_KEY = os.environ.get("LOCAL_API_KEY", "EMPTY")

declared_models = discover_declared_models(AGENT_DIR)
assert len(declared_models) == 1, f"Expected 1 declared model, got: {sorted(declared_models)}"
declared_model = next(iter(declared_models))
assert declared_model in ALLOWED_MODEL_NAMES, f"Unsupported model: {declared_model}"

MODEL_PATH = Path(f"/kaggle/input/models/google/gemma-4/other/{declared_model}/2")
assert MODEL_PATH.exists(), f"Model weights not found at {MODEL_PATH}."

adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
adapter_map = getattr(adapters, "adapters", None) or {}
for adapter_name in adapter_map:
    assert _SAFE_PATH_COMPONENT.fullmatch(adapter_name), f"Unsafe adapter name: {adapter_name}"
detected_ranks = [
    r for info in adapter_map.values() if (r := getattr(info, "rank", None)) is not None and r > 0
]

gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)

if "server_instance" in globals() and server_instance is not None:
    server_instance.stop()

vllm_cfg = VllmConfig(
    model=str(MODEL_PATH),
    port=8000,
    host="127.0.0.1",
    tool_call_parser="gemma4",
    reasoning_parser="gemma4",
    max_model_len=32768,
    dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto",
    gpu_memory_utilization=0.90,
    enable_auto_tool_choice=True,
    enable_lora=bool(adapter_map),
    max_loras=max(1, len(adapter_map)),
    max_lora_rank=max(detected_ranks) if detected_ranks else 128,
    scheduler_reserve_full_isl=False,
    tensor_parallel_size=tp_size,
    startup_timeout=60 * 20,
)
server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)
server_instance.start()
print(f"[✓] vLLM server started on {server_instance.base_url} (tp={tp_size}, adapters={list(adapter_map.keys())})")

models = server_instance.create_model_registry(
    aliases=[declared_model, *list(adapter_map.keys())],
    model_prefix="openai/",
    api_key=INFERENCE_API_KEY,
)

# %% [markdown]
# ## Cell 4: Controlled 2-Rung A/B Shootout on `DIAGNOSTIC_20` at `CONCURRENCY = 2`
# Evaluates `C1a_31B_tb1024` vs `C1b_31B_tb1024_direct` on the **exact same 20 tasks** (`10 fastapi` + `7 rich` + `3 requests`) with `CONCURRENCY = 2` so `tp=4` GPUs are never 4-way contended, and computes True `FAIL_TO_PASS` on `psf/requests`.

# %%
import asyncio
import concurrent.futures
import json
import re
import time
import pandas as pd
import yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator


def run_sync(coro_fn):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro_fn())).result()
    return asyncio.run(coro_fn())


# ==================== STAGE 2 SHOOTOUT CONTROLS ====================
# Evaluates C2_31B_F6_Apex_v3 on the exact same 20 tasks (DIAGNOSTIC_20, ~50 min)
# directly against yesterday's C1b_31B_tb1024_direct anchor (5/20 True-LB, 3/20 strict, 12/20 patches)!
# (For an even faster ~22 min check, change RUN_MODE = "CANARY_8".)
RUN_MODE = "DIAGNOSTIC_20"
CANDIDATES_TO_TEST = ["C2_31B_F6_Apex_v3"]
CONCURRENCY = 2
# ===================================================================

SMOKE_4_IDS = ["fastapi_14786", "rich_3518", "requests_6644", "httpx_3672"]
CUSTOM_3_IDS = ["fastapi_15588", "fastapi_14964", "requests_7433"]
CANARY_8_IDS = [
    "fastapi_14964", "fastapi_15588", "fastapi_15023",
    "rich_4076", "rich_4006", "rich_4075",
    "requests_7433", "requests_7505",
]

if RUN_MODE == "DIAGNOSTIC_20":
    remaining = [t for t in tasks if t.instance_id not in set(SMOKE_4_IDS)]
    SAMPLE_TASKS = (
        [t for t in remaining if "fastapi" in t.repo][:10]
        + [t for t in remaining if "rich" in t.repo][:7]
        + [t for t in remaining if "requests" in t.repo][:3]
    )
elif RUN_MODE == "CANARY_8":
    SAMPLE_TASKS = [task_map[tid] for tid in CANARY_8_IDS if tid in task_map]
elif RUN_MODE == "CUSTOM_3":
    SAMPLE_TASKS = [task_map[tid] for tid in CUSTOM_3_IDS if tid in task_map]
elif RUN_MODE == "SMOKE_4":
    SAMPLE_TASKS = [task_map[tid] for tid in SMOKE_4_IDS if tid in task_map]
else:
    SAMPLE_TASKS = tasks[:10]


def _check_true_f2p_for_requests(task, patch: str, tlog: str, raw_resolved: bool) -> bool:
    """
    In Kaggle Notebook subprocess mode, _MiniHttpbinHandler lacks /digest-auth and /delay,
    causing unrelated tests in test_requests.py to fail even on the 100% gold patch.
    Verify that all task-specific FAIL_TO_PASS nodes passed without collection errors.
    """
    if raw_resolved:
        return True
    if "requests" not in task.repo.lower() or not patch.strip() or not tlog.strip():
        return raw_resolved

    f2p = getattr(task, "fail_to_pass", None) or getattr(task, "FAIL_TO_PASS", None) or []
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except Exception:
            f2p = [f2p]
    if not f2p:
        return raw_resolved

    tlog_low = tlog.lower()
    if "error collecting" in tlog_low or "syntaxerror" in tlog_low or " passed" not in tlog_low:
        return False

    failed_lines = [
        ln for ln in tlog.splitlines()
        if ln.strip().startswith(("FAILED ", "ERROR ")) or "::" in ln and ("FAILED" in ln or "ERROR" in ln)
    ]
    for node in f2p:
        short_name = str(node).split("::")[-1].strip()
        if not short_name:
            continue
        for fln in failed_lines:
            if str(node) in fln or re.search(rf"\b{re.escape(short_name)}\b", fln):
                return False
    return True


async def _evaluate_candidate_async(cand_name: str, cand_dir: Path) -> dict:
    raw_eval_cfg = yaml.safe_load((cand_dir / "eval_config.yaml").read_text(encoding="utf-8")) or {}
    eval_section = raw_eval_cfg.get("evaluation", raw_eval_cfg)
    timeout_seconds = int(eval_section.get("timeout_seconds", 180))
    verification_timeout_seconds = int(eval_section.get("verification_timeout_seconds", 300))
    max_tool_calls = int(eval_section.get("max_tool_calls", 40))
    max_time_minutes = float(eval_section.get("max_time_minutes", 6.0))
    max_turns = int(eval_section.get("max_turns", 55))

    cand_adapters = discover_adapters(str(cand_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    res_dir = WORKING_DIR / f"results_{RUN_MODE.lower()}_{cand_name}"
    if res_dir.exists():
        shutil.rmtree(res_dir)
    p_dir = res_dir / "patches"
    t_dir = res_dir / "test_outputs"
    p_dir.mkdir(parents=True, exist_ok=True)
    t_dir.mkdir(parents=True, exist_ok=True)
    res_jsonl = res_dir / "task_results.jsonl"

    limits, gen_constraints = build_submission_limits()
    ecfg = EvalConfig(
        tasks_path=TASKS_PATH,
        snapshots_dir=DATA_DIR / "snapshots",
        results_dir=res_dir,
        submission_dir=cand_dir,
        models=models,
        sandbox="subprocess",
        timeout_seconds=timeout_seconds,
        verification_timeout_seconds=verification_timeout_seconds,
        max_time_minutes=max_time_minutes,
        max_tool_calls=max_tool_calls,
        max_turns=max_turns,
        limits=limits,
        generation_constraints=gen_constraints,
        adapter_manifest=cand_adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15,
            overlap_size=2,
            token_threshold=14336,
            event_retention_size=5,
        ),
        graph_dir=GRAPH_DIR,
        embeddings_dir=EMBEDDINGS_DIR,
        wheels_dir=DATA_DIR / "wheels",
        verbose=False,
    )
    evaluator = Evaluator(ecfg)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    done_cnt = 0
    res_cnt = 0
    adj_res_cnt = 0
    non_empty_patches = 0
    total_calls = 0
    lock = asyncio.Lock()

    print(f"\n==================== SHOOTOUT: {cand_name} ({len(SAMPLE_TASKS)} tasks | concurrency={CONCURRENCY}) ====================")

    async def _run_one(idx: int, task):
        nonlocal done_cnt, res_cnt, adj_res_cnt, non_empty_patches, total_calls
        async with sem:
            result = await evaluator.evaluate_task(task=task, task_index=idx, total_tasks=len(SAMPLE_TASKS))
        patch = (result.agent_patch or "").strip()
        (p_dir / f"{task.instance_id}.patch").write_text(patch + "\n", encoding="utf-8")

        if hasattr(result, "model_dump"):
            rdict = result.model_dump(mode="json", exclude={"trace", "session_trace"}, fallback=str)
        else:
            rdict = {k: str(v) for k, v in vars(result).items() if k not in ("trace", "session_trace")}
        rdict["instance_id"] = task.instance_id
        rdict["repo"] = task.repo

        tlog = str(rdict.get("test_output") or rdict.get("verification_output") or rdict.get("error") or "")
        (t_dir / f"{task.instance_id}.log").write_text(tlog, encoding="utf-8")

        adj_ok = _check_true_f2p_for_requests(task, patch, tlog, bool(result.resolved))
        rdict["adj_resolved"] = adj_ok

        async with lock:
            done_cnt += 1
            if result.resolved:
                res_cnt += 1
            if adj_ok:
                adj_res_cnt += 1
            if patch:
                non_empty_patches += 1
            total_calls += int(result.tool_calls or 0)
            with open(res_jsonl, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(rdict, default=str) + "\n")
            tag = "True (req-F2P)" if (adj_ok and not result.resolved) else str(bool(result.resolved))
            print(
                f"  [{done_cnt:2d}/{len(SAMPLE_TASKS)}] {task.instance_id:18s} | resolved={tag:14s} "
                f"| calls={int(result.tool_calls or 0):2d} | patch={len(patch):5d}c | {result.duration_seconds:5.1f}s "
                f"| err={result.error}"
            )

    await asyncio.gather(*[_run_one(i, t) for i, t in enumerate(SAMPLE_TASKS, start=1)])
    elapsed_min = round((time.time() - t0) / 60.0, 2)
    summary = {
        "candidate": cand_name,
        "run_mode": RUN_MODE,
        "concurrency": CONCURRENCY,
        "tasks": len(SAMPLE_TASKS),
        "strict_resolved": res_cnt,
        "true_lb_resolved": adj_res_cnt,
        "true_lb_pct": round(100.0 * adj_res_cnt / max(1, len(SAMPLE_TASKS)), 2),
        "non_empty_patches": non_empty_patches,
        "avg_tool_calls": round(total_calls / max(1, len(SAMPLE_TASKS)), 1),
        "elapsed_min": elapsed_min,
        "results_dir": str(res_dir),
    }
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[✓] FINISHED {cand_name}: true_lb={adj_res_cnt}/{len(SAMPLE_TASKS)} ({summary['true_lb_pct']}%) "
        f"[raw={res_cnt}] | patches={non_empty_patches}/{len(SAMPLE_TASKS)} | avg_calls={summary['avg_tool_calls']} | {elapsed_min} min"
    )
    return summary


shootout_rows = []
for cname in CANDIDATES_TO_TEST:
    assert cname in CANDIDATE_DIRS, f"Unknown candidate: {cname}"
    row = run_sync(lambda c=cname: _evaluate_candidate_async(c, CANDIDATE_DIRS[c]))
    shootout_rows.append(row)

shootout_df = pd.DataFrame(shootout_rows).sort_values(
    by=["true_lb_resolved", "strict_resolved", "non_empty_patches", "avg_tool_calls"],
    ascending=[False, False, False, True],
).reset_index(drop=True)

CHAMPION_NAME = str(shootout_df.loc[0, "candidate"])
RESULTS_DIR = Path(str(shootout_df.loc[0, "results_dir"]))
PATCHES_DIR = RESULTS_DIR / "patches"
TEST_OUT_DIR = RESULTS_DIR / "test_outputs"
RESULTS_JSONL = RESULTS_DIR / "task_results.jsonl"

print(f"\n[👑] SHOOTOUT WINNER: {CHAMPION_NAME} (Results in {RESULTS_DIR})")
display(shootout_df)

# %% [markdown]
# ## Cell 5: Upgraded Doctor Diagnostic Table & Auto-Generated `/kaggle/working/doctor_winner_dump.txt` (`< 1.5 MB`)
# Prints head-to-head results vs. Monday's `B3` anchor (`4/20`), inspects `SubAg` & `1stEd` turns, and writes a plain-text `/kaggle/working/doctor_winner_dump.txt` (`< 1.5 MB` hard-capped) ready for direct upload into Project Chat!

# %%
import json

B3_SOLVED_ANCHOR = ["rich_4077", "rich_3718", "requests_7433", "requests_7505"]
C1B_SOLVED_ANCHOR = ["fastapi_14964", "rich_4070", "rich_4077", "requests_7433", "requests_7502"]
print("=== PER-CANDIDATE SOLVED TASKS COMPARISON ===")
print(f"  {'B3_C1_budget40 (0.08 LB)':24s} ->  4/20 solved: {B3_SOLVED_ANCHOR} (8/20 patches @ CONCURRENCY=4)")
print(f"  {'C1b_31B_tb1024_direct':24s} ->  5/20 solved: {C1B_SOLVED_ANCHOR} (3 strict, 12/20 patches @ CONCURRENCY=2)")

all_cand_rows = {}
for cname in CANDIDATES_TO_TEST:
    c_res_dir = WORKING_DIR / f"results_{RUN_MODE.lower()}_{cname}"
    r_jf = c_res_dir / "task_results.jsonl"
    if r_jf.exists():
        rows = [json.loads(l) for l in r_jf.read_text(encoding="utf-8").splitlines() if l.strip()]
        all_cand_rows[cname] = {r["instance_id"]: r for r in rows}
        solved = [r["instance_id"] for r in rows if r.get("adj_resolved") or str(r.get("resolved")) == "True"]
        p_cnt = sum(
            1 for r in rows
            if (c_res_dir / "patches" / f"{r['instance_id']}.patch").exists()
            and (c_res_dir / "patches" / f"{r['instance_id']}.patch").read_text(encoding="utf-8").strip()
        )
        print(f"  {cname:24s} -> {len(solved):2d}/{len(rows)} solved: {solved} ({p_cnt}/{len(rows)} patches @ CONCURRENCY={CONCURRENCY})")


def _summarize_trace_file(tf: Path) -> dict:
    raw = json.loads(tf.read_text(encoding="utf-8")) if tf.exists() else {}
    steps = raw.get("steps", [])
    agent_steps = [s for s in steps if s.get("source") == "agent"]
    edit_fails = 0
    empty_cutoffs = 0
    subagent_calls = 0
    first_edit_step = "-"
    compact_lines = []

    for idx, s in enumerate(steps):
        src = str(s.get("source", "?"))
        ts = str(s.get("timestamp", ""))[-12:]
        calls = s.get("tool_calls") or []
        obs_str = str((s.get("observation") or {}).get("content", ""))
        is_err = any(k in obs_str for k in ('"status": "error"', '"error":', "Traceback", "ImportError", "SyntaxError"))
        obs_flag = "ERR" if is_err else "OK"

        if src == "agent" and not calls:
            empty_cutoffs += 1

        call_parts = []
        for c in calls:
            fn = str(c.get("function_name", "?"))
            args_preview = re.sub(r"\s+", " ", str(c.get("arguments", "")))[:150]
            call_parts.append(f"{fn}({args_preview})")
            if "analyzer" in fn.lower():
                subagent_calls += 1
            if fn in ("edit_file", "write_file") and first_edit_step == "-":
                first_edit_step = str(idx)
            if fn == "edit_file" and ('"status": "error"' in obs_str or '"error":' in obs_str):
                edit_fails += 1

        call_summary = ", ".join(call_parts) if call_parts else "NO_TOOL_CALL"
        obs_clean = re.sub(r"\s+", " ", obs_str).strip()
        obs_preview = (obs_clean[:160] + " ... " + obs_clean[-100:]) if len(obs_clean) > 280 else obs_clean
        compact_lines.append(
            f"  [{idx:02d}] {src:6s} | {ts:12s} | {call_summary} | obs={obs_flag}({len(obs_str)}c): {obs_preview}"
        )

    empty_cutoffs = max(0, empty_cutoffs - 1)
    return {
        "steps": len(steps),
        "subagent_calls": subagent_calls,
        "first_edit_step": first_edit_step,
        "edit_fails": edit_fails,
        "empty_cutoffs": empty_cutoffs,
        "compact_lines": compact_lines,
    }


results_by_id = all_cand_rows.get(CHAMPION_NAME, {})
trace_files = sorted((RESULTS_DIR / "traces").glob("trace_*.json")) if (RESULTS_DIR / "traces").exists() else []

print(f"\nDiagnostic Table for Winner [{CHAMPION_NAME}] -> {RESULTS_DIR} ({len(trace_files)} traces):\n")
header = (
    f"{'Task ID':18s} | {'TrueLB':7s} | {'Exit':4s} | {'Steps':5s} | "
    f"{'SubAg':5s} | {'1stEd':5s} | {'EditFail':8s} | {'Cutoffs':7s} | {'Patch':7s}"
)
print(header)
print("-" * len(header))

winner_table_lines = [header, "-" * len(header)]
winner_trace_summaries = {}

for tf in trace_files:
    tid = tf.stem.replace("trace_", "")
    tinfo = _summarize_trace_file(tf)
    winner_trace_summaries[tid] = tinfo
    rmeta = results_by_id.get(tid, {})
    patch_file = PATCHES_DIR / f"{tid}.patch"
    patch_len = len(patch_file.read_text(encoding="utf-8").strip()) if patch_file.exists() else 0
    line = (
        f"{tid:18s} | {str(rmeta.get('adj_resolved', rmeta.get('resolved', '?'))):7s} | "
        f"{str(rmeta.get('test_exit_code', '?')):4s} | {tinfo['steps']:5d} | "
        f"{tinfo['subagent_calls']:5d} | {tinfo['first_edit_step']:>5s} | "
        f"{tinfo['edit_fails']:8d} | {tinfo['empty_cutoffs']:7d} | {patch_len:6d}c"
    )
    print(line)
    winner_table_lines.append(line)

# --- Build plain-text /kaggle/working/doctor_winner_dump.txt (< 1.5 MB guaranteed) ---
MAX_DUMP_BYTES = 1_420_000
dump_sections = []

# 1. Shootout Overview & summary.json for all tested candidates
dump_sections.append(f"=== STAGE 1 SHOOTOUT OVERVIEW (WINNER: {CHAMPION_NAME}) ===")
dump_sections.append(shootout_df.to_string(index=False))
for cname in CANDIDATES_TO_TEST:
    s_file = WORKING_DIR / f"results_{RUN_MODE.lower()}_{cname}" / "summary.json"
    if s_file.exists():
        dump_sections.append(f"\n--- summary.json [{cname}] ---\n" + s_file.read_text(encoding="utf-8").strip())

# 2. Winner Diagnostic Table + Clean task_results.jsonl (excluding duplicate multi-MB raw test_output)
dump_sections.append(f"\n=== WINNER DIAGNOSTIC TABLE [{CHAMPION_NAME}] ===")
dump_sections.append("\n".join(winner_table_lines))

dump_sections.append(f"\n=== WINNER task_results.jsonl (clean metadata) [{CHAMPION_NAME}] ===")
for tid in sorted(results_by_id.keys()):
    clean_row = {
        k: v for k, v in results_by_id[tid].items()
        if k not in ("test_output", "verification_output")
    }
    p_file = PATCHES_DIR / f"{tid}.patch"
    clean_row["patch_bytes"] = len(p_file.read_text(encoding="utf-8").strip()) if p_file.exists() else 0
    dump_sections.append(json.dumps(clean_row, default=str))

# Side-by-side comparison rows for runner-up candidates
runner_ups = [c for c in CANDIDATES_TO_TEST if c != CHAMPION_NAME]
for rc in runner_ups:
    rc_dir = WORKING_DIR / f"results_{RUN_MODE.lower()}_{rc}"
    rc_map = all_cand_rows.get(rc, {})
    dump_sections.append(f"\n=== RUNNER-UP COMPACT TASK SUMMARY [{rc}] ===")
    for tid in sorted(rc_map.keys()):
        r = rc_map[tid]
        rf = rc_dir / "traces" / f"trace_{tid}.json"
        rt = _summarize_trace_file(rf) if rf.exists() else {"steps": 0, "subagent_calls": 0, "first_edit_step": "-"}
        rp = rc_dir / "patches" / f"{tid}.patch"
        rplen = len(rp.read_text(encoding="utf-8").strip()) if rp.exists() else 0
        dump_sections.append(
            f"  {tid:18s} | adj_resolved={str(r.get('adj_resolved')):5s} | exit={str(r.get('test_exit_code')):4s} "
            f"| calls={str(r.get('tool_calls')):3s} | steps={rt['steps']:2d} | subAg={rt['subagent_calls']} "
            f"| 1stEd={rt['first_edit_step']:>2s} | patch={rplen:5d}c | dur={r.get('duration_seconds')}s | err={r.get('error')}"
        )

# 3. Winner Compact Step-by-Step Traces (all 20 tasks)
dump_sections.append(f"\n=== WINNER COMPACT STEP TRACES [{CHAMPION_NAME}] ===")
for tid in sorted(winner_trace_summaries.keys()):
    tinfo = winner_trace_summaries[tid]
    rmeta = results_by_id.get(tid, {})
    dump_sections.append(
        f"\n--- TRACE {tid} (adj_resolved={rmeta.get('adj_resolved')}, exit={rmeta.get('test_exit_code')}, "
        f"dur={rmeta.get('duration_seconds')}s, steps={tinfo['steps']}, subAg={tinfo['subagent_calls']}, "
        f"1stEd={tinfo['first_edit_step']}) ---"
    )
    dump_sections.extend(tinfo["compact_lines"][:65])

# 4. Winner Patches (capped at 6000 chars per patch)
dump_sections.append(f"\n=== WINNER PATCHES [{CHAMPION_NAME}] ===")
for pf in sorted(PATCHES_DIR.glob("*.patch")):
    ptxt = pf.read_text(encoding="utf-8", errors="replace").strip()
    dump_sections.append(f"\n--- PATCH {pf.stem} ({len(ptxt)}c) ---")
    if ptxt:
        dump_sections.append(ptxt[:6000] + ("\n... [patch truncated]" if len(ptxt) > 6000 else ""))
    else:
        dump_sections.append("(EMPTY 0c PATCH)")

# 5. Winner Pytest Logs (head 500c + tail 1800c per task to catch both ImportError & FAIL_TO_PASS)
dump_sections.append(f"\n=== WINNER PYTEST HEAD+TAIL LOGS [{CHAMPION_NAME}] ===")
for lf in sorted(TEST_OUT_DIR.glob("*.log")):
    ltxt = lf.read_text(encoding="utf-8", errors="replace").strip()
    if not ltxt:
        dump_sections.append(f"\n--- LOG {lf.stem}: (EMPTY LOG / TIMEOUT BEFORE VERIFY) ---")
        continue
    snippet = ltxt if len(ltxt) <= 2400 else (ltxt[:500] + "\n... [middle omitted] ...\n" + ltxt[-1800:])
    dump_sections.append(f"\n--- LOG {lf.stem} ({len(ltxt)}c total) ---\n{snippet}")

full_dump_text = "\n".join(dump_sections)
encoded_dump = full_dump_text.encode("utf-8", errors="replace")
if len(encoded_dump) > MAX_DUMP_BYTES:
    full_dump_text = encoded_dump[:MAX_DUMP_BYTES].decode("utf-8", errors="ignore") + "\n\n[TRUNCATED AT 1.42 MB CAP]\n"

dump_txt_path = WORKING_DIR / "doctor_winner_dump.txt"
dump_txt_path.write_text(full_dump_text, encoding="utf-8")
dump_size_bytes = dump_txt_path.stat().st_size
assert dump_size_bytes < 1_500_000, f"doctor_winner_dump.txt exceeded 1.5 MB: {dump_size_bytes}"
print(f"\n[✓] Wrote chat-ready plain-text dump: {dump_txt_path} ({dump_size_bytes / 1024:.1f} KB — < 1.5 MB verified!)")

doctor_zip = shutil.make_archive(
    str(WORKING_DIR / f"doctor_{CHAMPION_NAME}_{RUN_MODE.lower()}"),
    "zip",
    root_dir=RESULTS_DIR,
)
print(f"[✓] Archived full winner directory backup: {doctor_zip} ({Path(doctor_zip).stat().st_size / 1024:.1f} KB)")

# %% [markdown]
# ## Cell 6: Package & Strictly Validate Final Competition `/kaggle/working/submission.zip`
# Automatically packages `CHAMPION_NAME` (or override `SELECTED_SUBMISSION` explicitly) into `/kaggle/working/submission.zip` — **NO trailing cells exist below this cell!**

# %%
import os
import shutil
import zipfile
from pathlib import Path
from IPython.display import FileLink, display
from adk_submission import validate_directory
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_SUBMISSION_EXTENSIONS, MAX_SUBMISSION_SIZE_BYTES, build_submission_limits

# Automatically select the empirical winner between C2_31B_F6_Apex_v3 and C1b_31B_tb1024_direct!
if "shootout_df" in globals() and len(shootout_df) > 0 and RUN_MODE == "DIAGNOSTIC_20":
    top_row = shootout_df.iloc[0]
    c2_key = (int(top_row["true_lb_resolved"]), int(top_row["strict_resolved"]), int(top_row["non_empty_patches"]))
    c1b_anchor_key = (5, 3, 12)  # Yesterday's exact C1b_31B_tb1024_direct DIAGNOSTIC_20 score
    if str(top_row["candidate"]) == "C2_31B_F6_Apex_v3" and c2_key < c1b_anchor_key:
        print(f"[AUTO-GUARD] C2 scored {c2_key} < C1b anchor {c1b_anchor_key}; locking proven C1b_31B_tb1024_direct!")
        SELECTED_SUBMISSION = "C1b_31B_tb1024_direct"
    else:
        print(f"[AUTO-GUARD] Locking new champion {top_row['candidate']} with score {c2_key} >= C1b anchor {c1b_anchor_key}!")
        SELECTED_SUBMISSION = str(top_row["candidate"])
else:
    SELECTED_SUBMISSION = globals().get("CHAMPION_NAME", "C2_31B_F6_Apex_v3")
assert SELECTED_SUBMISSION in CANDIDATE_DIRS, f"Unknown candidate: {SELECTED_SUBMISSION}"

if AGENT_DIR.exists():
    shutil.rmtree(AGENT_DIR)
shutil.copytree(CANDIDATE_DIRS[SELECTED_SUBMISSION], AGENT_DIR)

# 1. Symlink & Single-Model Check
for dirpath, dirnames, filenames in os.walk(AGENT_DIR, followlinks=False):
    parent = Path(dirpath)
    for name in (*dirnames, *filenames):
        assert not (parent / name).is_symlink(), f"Symlink forbidden: {(parent / name).relative_to(AGENT_DIR)}"

declared = discover_declared_models(AGENT_DIR)
assert declared == {MODEL_NAME}, f"Invalid declared models: {declared} (expected {MODEL_NAME})"

# 2. Official ADK Declarative Directory Validation
limits, _ = build_submission_limits()
validate_directory(AGENT_DIR, limits=limits)

# 3. Package /kaggle/working/submission.zip
zip_out = WORKING_DIR / "submission.zip"
if zip_out.exists():
    zip_out.unlink()
zip_path = Path(shutil.make_archive(str(WORKING_DIR / "submission"), "zip", root_dir=AGENT_DIR))

with zipfile.ZipFile(zip_path, "r") as zf:
    infos = [i for i in zf.infolist() if not i.is_dir()]
    assert "agent.yaml" in [i.filename for i in infos], "agent.yaml must be at archive root!"
    total_size = sum(i.file_size for i in infos)
    assert total_size <= MAX_SUBMISSION_SIZE_BYTES, f"Archive exceeds 3 GiB limit: {total_size}"
    print(f"Packaged [{SELECTED_SUBMISSION}] -> {zip_path}:")
    for info in sorted(infos, key=lambda x: x.filename):
        ext = Path(info.filename).suffix.lower()
        assert ext in ALLOWED_SUBMISSION_EXTENSIONS, f"Disallowed extension: {info.filename}"
        print(f"  - {info.filename:52s} ({info.file_size / 1024:7.2f} KB)")

print(f"\n[✓] READY FOR KAGGLE SUBMISSION: {zip_path} ({zip_path.stat().st_size / 1024:.2f} KB)")
os.chdir(WORKING_DIR)
display(FileLink("submission.zip"))


# %%




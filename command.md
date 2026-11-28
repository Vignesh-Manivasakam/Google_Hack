Read `FINAL_KAGGLE_QAT_PLAN.md` and `Review.md` completely before writing any code. Follow Section 4 (Anti-Skip Checklist) strictly — do NOT skip or simplify any requirement.

Using `train_e4b_v6_local.py`, `parity_common.py`, and `kaggle_bakeoff_e4b.ipynb` in this folder, create TWO clean, self-contained Kaggle notebooks:

1. `kaggle_train_qat_v6_and_v1warm.ipynb` (runs on Kaggle GPU using `gemma-4-e4b-it-qat-q4_0-unquantized` — NEVER vanilla `gemma-4-e4b-it`):
   - Trains **Option 2 (`v6_qat`)**: 60 tensors (`L22` q/k/v/o, `L23` all 7, `L24/26` q/o, `L28/30/36` q/o+MLP), `r=8`, `alpha=16`, `lr=1e-4`, `kl_beta=0.10`, `1.0` epoch on `V6_Data` (336 train / 36 val).
   - Trains **Option 3 (`v1_warm_qat`)**: warm-starts from `submission_E4B_MultiLoRA/adapters/lora_v1` (54 tensors), `lr=5e-5`, `kl_beta=0.12`, `1.0` epoch on `V6_Data`.
   - MUST preserve exact `train_e4b_v6_local.py` logic: native BF16 + PyTorch SDPA (NO NF4, NO FA2), `tools=KAGGLE_TOOLS`, `loss_mask=0` zeroed on synthetic bad edits, `W_SUBMIT=1.0`, `W_EDIT=1.5`, ID `50` kept as assistant stop token.
   - Runs `pack_kaggle_bakeoff.py` G0 assertions on both exported adapters (`base_model_name_or_path == "google/gemma-4-e4b-it-qat-w4a16-ct"`, explicit `target_modules` list, `r=8`, `0` `embed_tokens`/`lm_head` keys) and saves them to `/kaggle/working/qat_adapters/`.

2. `kaggle_bakeoff_qat_final.ipynb` (cloned from `kaggle_bakeoff_e4b.ipynb`):
   - Evaluates 4 arms on `key10` with `DEDUP_ON = False`, `LEGACY_PROMPT_TOOLS = True`, `frequency_penalty = 0.30`, `thinking_budget = 1024`, `include_thoughts = False`:
     (1) `lora_v1` (control), (2) `base_qat_nolora` (Option 1: no LoRA + `system_v6_grep.md`), (3) `v6_qat` (Option 2), (4) `v1_warm_qat` (Option 3).
   - Supports `EVAL_MODE = "holdout30"` on the single `key10` winner with the `>= 7/30` & `0` collapse promote/stop gate.

Verify both notebooks parse as valid JSON + valid Python syntax in every cell, and print a line-by-line confirmation of Checks 1–5 from `FINAL_KAGGLE_QAT_PLAN.md` Section 4.

python -u win_frontier_6harness_overnight.py --factory --candidates E4B_LoRA_v1 --model_id ..\gemma-4-e4b-it --adapter_dir adapters\e4b_qat_lora --tasks_path tasks.jsonl --task_mode train99 --workers 1 --batch_size 1 --out_dir ..\e4b_lora_v1_factory 


  python generate_onpolicy_passes.py --results_dir ..\e4b_lora_v1_factory --out_jsonl onpolicy_lora_v1_passing.jsonl --max_per_task 2
python generate_aggrevate_prefixes.py --results_dir ..\e4b_lora_v1_factory --out_jsonl aggrevate_failure_prefixes.jsonl --target_count 28

"""
(GH) PS C:\Users\mqg1cob\Documents\Google Hack\gemma-4-developer-agent> python -u win_frontier_6harness_overnight.py --factory --candidates E4B_LoRA_v1 --model_id ..\gemma-4-e4b-it --adapter_dir adapters\e4b_qat_lora --tasks_path tasks.jsonl --task_mode train99 --workers 1 --batch_size 1 --out_dir ..\e4b_lora_v1_factory

[GPU] Initializing BatchedGemmaServer (..\gemma-4-e4b-it, batch_size=1)...
[PEFT-MOUNT] Targeting LoRA v1 adapter: C:\Users\mqg1cob\Documents\Google Hack\gemma-4-developer-agent\adapters\e4b_qat_lora
[+] Fast-loading ..\gemma-4-e4b-it onto GPU 0 (hybrid PLE/multimodal CPU offload=True)...
C:\Users\mqg1cob\.conda\envs\GH\Lib\site-packages\transformers\modeling_utils.py:5231: UserWarning: expandable_segments not supported on this platform (Triggered internally at C:\bld\libtorch_1750199249093\work\c10/cuda/CUDAAllocatorConfig.h:28.)
  _ = torch.empty(int(byte_count // 2), dtype=torch.float16, device=device, requires_grad=False)
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2076/2076 [00:12<00:00, 172.01it/s]
[WARN] Hybrid device_map fallback (The device_map provided does not give any device for the following parameters: lm_head.weight); loading with device_map={'': 0}...
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2076/2076 [00:13<00:00, 159.45it/s]
[+] Verified CPU-pinned `embed_tokens_per_layer` (freed ~5.49 GiB GPU VRAM -> ~8.41 GiB active)!
[+] Attaching PEFT adapter from C:\Users\mqg1cob\Documents\Google Hack\gemma-4-developer-agent\adapters\e4b_qat_lora...
C:\Users\mqg1cob\.conda\envs\GH\Lib\site-packages\transformers\modeling_utils.py:3303: FutureWarning: `is_gradient_checkpointing` is deprecated and will be removed in v6. Use `is_activation_checkpointing` instead.
  warnings.warn(
[+] Multi-turn Gemma 4 chat template verified (mode=str_content)!
[PEFT] Successfully mounted adapter from C:\Users\mqg1cob\Documents\Google Hack\gemma-4-developer-agent\adapters\e4b_qat_lora (< 9 GiB VRAM footprint, zero NF4)!
[GPU] Ready! Active VRAM (language-only): 8.67 GiB / 16.00 GiB
[DATA] Loading benchmark tasks (all) from C:\Users\mqg1cob\Documents\Google Hack\gemma-4-developer-agent\tasks.jsonl...
[DATA] Loaded all 129 tasks from tasks.jsonl

==================== [ROLLOUT FACTORY LAUNCHED] ====================
  * Mode:             Dedicated Single-Pass Rollout Factory (Persistent Cache)
  * Candidates (1): ['E4B_LoRA_v1']
  * Total Tasks:      129 tasks
  * Output Directory: C:\Users\mqg1cob\Documents\Google Hack\e4b_lora_v1_factory
  * Model ID:         ..\gemma-4-e4b-it
  * Mounted LoRA:     C:\Users\mqg1cob\Documents\Google Hack\gemma-4-developer-agent\adapters\e4b_qat_lora (< 9 GiB VRAM, zero NF4)
====================================================================


>>> Executing Rollout Factory for Candidate: E4B_LoRA_v1 (129 tasks) <<<

==================== [E4B_LoRA_v1] (129 run + 0 cached = 129 tasks | workers=1) ====================
    [E4B_LoRA_v1            | fastapi_15661 ] s01 -> run_command(ls -R .github/workflows/)
    [E4B_LoRA_v1            | fastapi_15661 ] s03 -> run_command(ls -R .github/workflows/)
    [E4B_LoRA_v1            | fastapi_15661 ] s05 -> run_command(grep -r "release" .)
    [E4B_LoRA_v1            | fastapi_15661 ] s06 -> read_file(scripts/add_latest_release_date.py:1-100)
    [E4B_LoRA_v1            | fastapi_15661 ] s08 -> read_file(scripts/add_latest_release_date.py:1-40)
    [E4B_LoRA_v1            | fastapi_15661 ] s09 -> edit_file(scripts/add_latest_release_date.py)
    [E4B_LoRA_v1            | fastapi_15661 ] s10 -> edit_file(scripts/add_latest_release_date.py)
    [E4B_LoRA_v1            | fastapi_15661 ] s11 -> read_file(scripts/add_latest_release_date.py:7-10)
    [E4B_LoRA_v1            | fastapi_15661 ] s12 -> read_file(scripts/add_latest_release_date.py:7-15)
    [E4B_LoRA_v1            | fastapi_15661 ] s13 -> read_file(scripts/add_latest_release_date.py:8-9)
    [E4B_LoRA_v1            | fastapi_15661 ] s14 -> read_file(scripts/add_latest_release_date.py:8-9)
    [E4B_LoRA_v1            | fastapi_15661 ] s15 -> edit_file(scripts/add_latest_release_date.py)
    [E4B_LoRA_v1            | fastapi_15661 ] s16 -> edit_file(scripts/add_latest_release_date.py)
    [E4B_LoRA_v1            | fastapi_15661 ] s17 -> edit_file(scripts/add_latest_release_date.py)
    [E4B_LoRA_v1            | fastapi_15661 ] s18 -> edit_file(scripts/add_latest_release_date.py)
    [E4B_LoRA_v1            | fastapi_15661 ] s20 -> run_command(find tests -name "*release_notes*.py")
    [E4B_LoRA_v1            | fastapi_15661 ] s22 -> run_command(grep -r "release-notes.md" .)
    [E4B_LoRA_v1            | fastapi_15661 ] s24 -> run_command(PYTHONPATH=src:. python3 -m py_compile scripts/add_latest_release_date.py)
    [E4B_LoRA_v1            | fastapi_15661 ] s26 -> run_command(PYTHONPATH=src:. python3 -c "import scripts.add_latest_release_date; scripts.add)
    [E4B_LoRA_v1            | fastapi_15661 ] s28 -> run_command(git diff)
    [E4B_LoRA_v1            | fastapi_15661 ] s29 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s29 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s29 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s30 -> run_command(PYTHONPATH=src:. python3 -c "import scripts.add_latest_release_date; scripts.add)
    [E4B_LoRA_v1            | fastapi_15661 ] s32 -> run_command(PYTHONPATH=src:. python3 -c "import scripts.add_latest_release_date; scripts.add)
    [E4B_LoRA_v1            | fastapi_15661 ] s33 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s33 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s34 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s37 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s38 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s38 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15661 ] s39 -> submit_patch()
  [ 1/129] fastapi_15661    | PASS=False | sub=False | steps=40 | ok=0 | ef=6 | 0c
    [E4B_LoRA_v1            | fastapi_15588 ] s01 -> search_similar_code(Server Sent Event)
    [E4B_LoRA_v1            | fastapi_15588 ] s03 -> get_code_neighbors(fastapi.sse.ServerSentEvent)
    [E4B_LoRA_v1            | fastapi_15588 ] s05 -> read_file(fastapi/sse.py:1-150)
    [E4B_LoRA_v1            | fastapi_15588 ] s07 -> read_file(fastapi/sse.py:151-150)
    [E4B_LoRA_v1            | fastapi_15588 ] s09 -> read_file(fastapi/sse.py:151-222)
    [E4B_LoRA_v1            | fastapi_15588 ] s10 -> edit_file(fastapi/sse.py)
    [E4B_LoRA_v1            | fastapi_15588 ] s11 -> edit_file(fastapi/sse.py)
    [E4B_LoRA_v1            | fastapi_15588 ] s12 -> read_file(fastapi/sse.py:190-210)
    [E4B_LoRA_v1            | fastapi_15588 ] s13 -> read_file(fastapi/sse.py:190-210)
    [E4B_LoRA_v1            | fastapi_15588 ] s14 -> edit_file(fastapi/sse.py)
    [E4B_LoRA_v1            | fastapi_15588 ] s15 -> edit_file(fastapi/sse.py)
    [E4B_LoRA_v1            | fastapi_15588 ] s16 -> read_file(fastapi/sse.py:205-215)
    [E4B_LoRA_v1            | fastapi_15588 ] s17 -> read_file(fastapi/sse.py:208-215)
    [E4B_LoRA_v1            | fastapi_15588 ] s18 -> edit_file(fastapi/sse.py)
    [E4B_LoRA_v1            | fastapi_15588 ] s19 -> edit_file(fastapi/sse.py)
    [E4B_LoRA_v1            | fastapi_15588 ] s20 -> read_file(fastapi/sse.py:200-210)
    [E4B_LoRA_v1            | fastapi_15588 ] s21 -> read_file(fastapi/sse.py:200-210)
    [E4B_LoRA_v1            | fastapi_15588 ] s22 -> run_command(find tests -name "*sse*.py")
    [E4B_LoRA_v1            | fastapi_15588 ] s23 -> run_command(find tests -name "*sse*.py")
    [E4B_LoRA_v1            | fastapi_15588 ] s24 -> run_command(grep -r "ServerSentEvent" .)
    [E4B_LoRA_v1            | fastapi_15588 ] s25 -> run_command(grep -r "ServerSentEvent" tests/)
    [E4B_LoRA_v1            | fastapi_15588 ] s25 -> submit_patch()
    [E4B_LoRA_v1            | fastapi_15588 ] s26 -> run_command(PYTHONPATH=src:. pytest tests/test_sse.py -k "ServerSentEvent" -q --tb=short)
    [E4B_LoRA_v1            | fastapi_15588 ] s26 -> submit_patch()
  [ 2/129] fastapi_15588    | PASS=False | sub=True  | steps=26 | ok=3 | ef=3 | 826c
  """

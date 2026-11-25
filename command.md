cd "C:\Users\mqg1cob\Documents\Google Hack\gemma-4-developer-agent"
Expand-Archive -Path .\localdrop_v9.zip -DestinationPath . -Force

# 1. Instant 0-GPU dry-run verification (<10s)
python win_frontier_6harness_overnight.py --dry_run_test
python generate_onpolicy_passes.py --dry_run_test
python generate_aggrevate_prefixes.py --dry_run_test

# 2. Fast 4-task smoke4 parity check (verify no_call/task < 2.0 and no_anyio_bash_emissions == 0)
python -u win_frontier_6harness_overnight.py --factory --candidates E4B_Base_v1_5 --model_id ..\gemma-4-e4b-it --adapter_dir none --tasks_path tasks.jsonl --task_mode smoke4 --workers 1 --batch_size 1 --out_dir ..\e4b_v1_5_parity_smoke4
python -u win_frontier_6harness_overnight.py --factory --candidates E4B_LoRA_v1_5 --model_id ..\gemma-4-e4b-it --adapter_dir adapters\e4b_qat_lora --tasks_path tasks.jsonl --task_mode smoke4 --workers 1 --batch_size 1 --out_dir ..\e4b_v1_5_parity_smoke4

# 3. Clean Val20 Parity Gate (Base + 2048 B vs LoRA_v1 2149 B vs LoRA_v1.5 2048 B)
python -u win_frontier_6harness_overnight.py --factory --candidates E4B_Base_v1_5 --model_id ..\gemma-4-e4b-it --adapter_dir none --tasks_path tasks.jsonl --task_mode val20 --workers 1 --batch_size 1 --out_dir ..\e4b_v1_5_parity_val20
python -u win_frontier_6harness_overnight.py --factory --candidates E4B_LoRA_v1_2149B --model_id ..\gemma-4-e4b-it --adapter_dir adapters\e4b_qat_lora --tasks_path tasks.jsonl --task_mode val20 --workers 1 --batch_size 1 --out_dir ..\e4b_v1_5_parity_val20
python -u win_frontier_6harness_overnight.py --factory --candidates E4B_LoRA_v1_5 --model_id ..\gemma-4-e4b-it --adapter_dir adapters\e4b_qat_lora --tasks_path tasks.jsonl --task_mode val20 --workers 1 --batch_size 1 --out_dir ..\e4b_v1_5_parity_val20

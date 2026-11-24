python -u win_frontier_6harness_overnight.py `
  --factory `
  --candidates E4B_LoRA_v1 `
  --model_id ..\gemma-4-e4b-it `
  --adapter_dir adapters\e4b_qat_lora `
  --tasks_path tasks.jsonl `
  --task_mode train99 `
  --workers 1 `
  --batch_size 1 `
  --out_dir ..\e4b_factory_train99



```python generate_onpolicy_passes.py --results_dir ..\e4b_lora_v1_factory --out_jsonl onpolicy_lora_v1_passing.jsonl --splits splits_99_30.json --max_per_task 2```
```python generate_aggrevate_prefixes.py --results_dir ..\e4b_lora_v1_factory --out_jsonl aggrevate_failure_prefixes.jsonl --splits splits_99_30.json --target_count 28```

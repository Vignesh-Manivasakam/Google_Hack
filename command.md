# 1. LoRA_v1 (uses default submission_E4B_LoRA_v1 in Google Hack\gemma-4-developer-agent)
python viz_live_layers.py --task_ids fastapi_14786,fastapi_14077,fastapi_13786 --max_steps 12 --no_live

# 2. V3_Fixed Arm B comparison on the exact same 3 tasks
python viz_live_layers.py --submission_dir submission_E4B_LoRA_v3_ArmB --task_ids fastapi_14786,fastapi_14077,fastapi_13786 --max_steps 12 --no_live

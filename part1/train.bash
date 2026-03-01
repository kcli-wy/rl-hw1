#!/bin/bash


uv run src/scripts/run.py --exp_name trajectory_return
uv run src/scripts/run.py --use_reward_to_go --exp_name rtg

uv run src/scripts/run.py --use_reward_to_go  --exp_name no_baseline
uv run src/scripts/run.py --use_reward_to_go --baseline --exp_name baseline
uv run src/scripts/run.py --use_reward_to_go --baseline -na --exp_name adv_norm


uv run src/scripts/run.py --use_reward_to_go --use_baseline --gae_lambda 0      --exp_name lambda_0
uv run src/scripts/run.py --use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name lambda_0.95
uv run src/scripts/run.py --use_reward_to_go --use_baseline --gae_lambda 1      --exp_name lambda_1
 


# Part 1
```
cd part1
uv sync
```
## 1.1
```
uv run src/scripts/run.py --exp_name  trajectory_return
uv run src/scripts/run.py --use_reward_to_go --exp_name rtg
```

## 1.2
```
uv run src/scripts/run.py --use_reward_to_go  --exp_name no_baseline
uv run src/scripts/run.py --use_reward_to_go --use_baseline --exp_name baseline
uv run src/scripts/run.py --use_reward_to_go --use_baseline -na --exp_name adv_norm
 ```

 ## 1.3

 ```
 uv run src/scripts/run.py --use_reward_to_go --use_baseline --gae_lambda 0      --exp_name lambda_0 
uv run src/scripts/run.py --use_reward_to_go --use_baseline --gae_lambda 0.95 --exp_name lambda_0.95
uv run src/scripts/run.py --use_reward_to_go --use_baseline --gae_lambda 1      --exp_name lambda_1
```


# Part 2
```
cd part2
uv sync
```
## 2.1

```
uv run part2/src/scripts/run_dqn.py --config_file experiments/dqn/dqn.yaml --eval_interval 2500
```

## 2.2

```
uv run part2/src/scripts/run_dqn.py --config_file experiments/dqn/multi_step_dqn_1.yaml --eval_interval 2500
uv run part2/src/scripts/run_dqn.py --config_file experiments/dqn/multi_step_dqn_3.yaml --eval_interval 2500
uv run part2/src/scripts/run_dqn.py --config_file experiments/dqn/multi_step_dqn_5.yaml --eval_interval 2500
```

## 2.3

```
uv run part2/src/scripts/run_dqn.py --config_file experiments/dqn/double_dqn.yaml --eval_interval 2500
```
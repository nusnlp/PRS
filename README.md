## Preference Guided Reflective Sampling for Aligning Language Models

Code for our paper [Preference Guided Reflective Sampling for Aligning Language Models](). We provide the code for generating data with PReS, where the generated data can be used in offline RL training for aligning a language model. For comparison, we also include the baseline sampling methods such as random sampling.

## Quick Start
To sample responses, in file of run_best_of_N.eval_mode.sh, you have to specify the 1. dataset path, 2. the policy model and 3. the reward model, then run:
```bash
bash run_best_of_N.eval_mode.sh
```


## Preference Guided Reflective Sampling for Aligning Language Models

Code for our paper [Preference Guided Reflective Sampling for Aligning Language Models](). We provide the code for generating data with PReS, where the generated data can be used in offline RL training for aligning a language model. For comparison, we also include the baseline sampling methods such as random sampling.

## Quick Start
To sample responses, you have to specify the dataset path and the policy model in run_best_of_N.eval_mode.sh
```bash
bash run_best_of_N.eval_mode.sh
```


## Preference Guided Reflective Sampling for Aligning Language Models

Code for our paper [Preference Guided Reflective Sampling for Aligning Language Models](). We provide the code for generating data with *PReS*, where the generated data can be used in offline RL training for aligning a language model. For comparison, we also include the baseline sampling methods such as random sampling.

## Quick Start
To sample responses, in file of `run_best_of_N.eval_mode.sh`, you have to specify the 

1. data path of prompts;
2. the policy model, such as Mistral-7B-Instruct-v0.2;
3. the reward model, such as [UltraRM-13b](https://huggingface.co/openbmb/UltraRM-13b). 

Then run:
```bash
bash run_best_of_N.eval_mode.sh
```

For *PReS*, you will get two files of responses. You can combine them with `combine_for_tree_search.py`:

```bash
python combine_for_tree_search.py path_to_initial_response path_to_refinement path_to_save
```

## Requirements

1. Install `vLLM`: We use vLLM to fasten model sampling, so you have to install vLLM from [here](https://docs.vllm.ai/en/latest/getting_started/installation.html).


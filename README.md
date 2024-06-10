## Preference Guided Reflective Sampling for Aligning Language Models

Code for our paper [Preference Guided Reflective Sampling for Aligning Language Models](). We provide the code for generating data with *PReS*, where the generated data can be used in offline RL training for aligning a language model. For comparison, we also include the baseline sampling methods such as random sampling.


## Overview
We propose a new sampling method named *Preference Guided Reflective Sampling for Aligning Language Models (PReS)*. *PReS* frames the data generation as the optimization process to a specified user preference described in natural language, such that "Can you give me a concise response without explanations?". 

### Compare with Random Sampling (Rand)
**Preference**: *PReS* needs to specify a user preference for optimization but Rand does not. 
![](./figures/compare_random.png)

### Method
*PReS* employs a tree-based generation approach to optimize outputs aligned to user preference. 
<img src="./figures/method.png" height="300" alt="Description of Image">
![](./figures/method.png)






## Quick Start
To sample responses, in file of `run_best_of_N.eval_mode.sh`, you have to specify the 

1. data path of prompts;
2. the policy model, such as Mistral-7B-Instruct-v0.2;
3. the reward model, such as [UltraRM-13b](https://huggingface.co/openbmb/UltraRM-13b). 

We provide the example data of prompts from Alpaca (see `data/alpaca_gpt4.dev_set.num=100.w_preference_by_gpt-3.5.jsonl`).

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


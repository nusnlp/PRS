## PRS:
Code for our EMNLP2024 paper [Preference-Guided Reflective Sampling for Aligning Language Models](https://arxiv.org/abs/2408.12163). We provide the code for generating data with *PRS*, where the generated data can be used in iterative offline RL training for aligning a language model. For comparison, we also include the baseline sampling methods such as random sampling.

[Website](https://data-sampling-prs.github.io/)

## News:
- Oct. 2024: *PRS* is accepted by EMNLP2024 as the main paper!
- Jun. 2024: Release the first version of code for *PRS*.

## Quick Links
  - [Quick Start](#quick-start)
  - [Requirements](#requirements)

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

For *PRS*, you will get two files of responses. You can combine them with `combine_for_tree_search.py`:

```bash
python combine_for_tree_search.py path_to_initial_response path_to_refinement path_to_save
```

## Requirements

1. Install `vLLM`: We use vLLM to fasten model sampling, so you have to install vLLM from [here](https://docs.vllm.ai/en/latest/getting_started/installation.html).


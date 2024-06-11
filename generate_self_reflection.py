from vllm import LLM, SamplingParams
from utils.prompter import Prompter
import argparse
import jsonlines
from tqdm import tqdm
import torch
import json
import os
from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List

ultrarm_template= "Human: {instruction}\nAssistant: {response}"

class LlamaRewardModel(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.regression_head = nn.Linear(self.config.hidden_size, 1, bias=False)

    def forward( # args are the same as LlamaForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                inputs_embeds=inputs_embeds,
                            )

        hidden_states = transformer_outputs[0]
        rewards = self.regression_head(hidden_states).squeeze(-1)

        ends = attention_mask.cumsum(dim=1).argmax(dim=1).view(-1,1)
        rewards = torch.gather(rewards, 1, ends)

        return rewards


def save_data_to_json(strings, file_name):
    with open(file_name, 'a', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)

def evaluate_batch(args, model, prompter, instructions, inputs=None, N=1):
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=1,
                                     top_k=50,
                                     max_tokens=args.max_tokens,
                                     stop = '</s>',
                                     presence_penalty=1.2
                                     )

    if inputs is None:
        prompts = [prompter.generate_prompt(instruction, "") for instruction in instructions]
    else:
        prompts = [prompter.generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]

    print(prompts[0])
    print('\n\n------------\n\n')

    all_responses = [None] * len(instructions)

    for iii in range(N):
        print(f'round {iii+1} is going on ...')
        outputs = model.generate(prompts, sampling_params)
        outputs = [ (int(kk.request_id), kk)  for kk in outputs   ]
        sorted_list = sorted(outputs, key=lambda x: x[0])
        outputs = [x[1] for x in sorted_list]
        responses = [response.outputs[0].text for response in outputs]
        
        print(responses[-1])

        for i, response in enumerate(responses):
            if all_responses[i] == None:
                all_responses[i] = []
            all_responses[i].append(response)

    return all_responses


def main(
):
    parser = argparse.ArgumentParser(prog='Generate', description='Generate responses on the eval set')
    parser.add_argument('-o,', '--output', required=True, help="File path to the output responses")
    parser.add_argument('--base_model', required=True, help="the base model")
    parser.add_argument('--reward_model', required=False, help="the reward model")
    parser.add_argument('--prompt_template', required=True, help="path to prompt template")
    parser.add_argument('--input', type=str, required=False, help="")
    parser.add_argument('--max_tokens', type=int, default=1024, required=False, help="")
    parser.add_argument('--seed', type=int, default=0, required=False, help="")
    parser.add_argument('--path_to_gen_self_feedback', type=str, required=True, help="")
    parser.add_argument('--path_to_revise_w_feedback', type=str, required=True, help="")
    parser.add_argument('--path_to_revise_wo_feedback', type=str, required=False, help="")
    parser.add_argument('--mode', choices=['sample_N', 'sample_N_wo_prefer', 'self_reflection', 'tree_search', 'sample_greedy_search', 'tree_search_wo_feedback'], required=True, help="")
    parser.add_argument('--n_sample', type=int, default=8, required=False, help="")
    parser.add_argument('--path_user_preference', type=str, required=False, help="")
    parser.add_argument('--batch_size', type=int, default=10, required=False, help="")
    

    args = parser.parse_args()

    if args.mode == 'sample_N':
        sample_N(args)
    elif args.mode == 'tree_search':
        tree_search(args)
    elif args.mode == 'tree_search_wo_feedback':
        tree_search_wo_feedback(args)




def sample_N(args):
    model = LLM(model=args.base_model, tensor_parallel_size=torch.cuda.device_count())
    prompter = Prompter(args.prompt_template)

    
    data_with_response = {}
    if os.path.exists(args.output):
        with jsonlines.open(args.output) as reader:
            for item in reader:
                data_with_response[item['id']] = ""
    print(data_with_response)


    root_user_preference = ''
    if args.path_user_preference and os.path.exists(args.path_user_preference):
        with open(args.path_user_preference, 'r') as f:
            root_user_preference = f.read().strip()

    instructions, all_items = [], []
    with open(args.input, 'r') as f:
        content = f.readlines()
        for item in content:
            item = json.loads(item)
            if item['id'] in data_with_response:
                continue
            if root_user_preference == 'preference_2':
                c_inst = item['instruction'] + '\n\n' + item['preference_2']
            elif root_user_preference == 'preference_1':
                c_inst = item['instruction'] + '\n\n' + item['preference_1']
            else:
                c_inst = item['instruction'] + '\n\n' + root_user_preference
            instructions.append(c_inst.strip())
            all_items.append(item)
    
    ### best of N
    for i in tqdm(range(0, len(all_items), args.batch_size)):
        c_instructions, c_items = [], []
        for j in range(args.batch_size):
            if i + j <= len(all_items)-1:
                c_instructions.append(instructions[i+j])
                c_items.append(all_items[i+j])

        c_all_responses = evaluate_batch(args, model, prompter, c_instructions, N=args.n_sample)
        for i, c_responses in enumerate(c_all_responses):
            c_items[i]['responses'] = c_responses
        ## save the data
        save_data_to_json(c_items, args.output)
    

def tree_search(args):
    model = LLM(model=args.base_model, tensor_parallel_size=torch.cuda.device_count())
    prompter = Prompter(args.prompt_template)

    with open(args.path_to_gen_self_feedback, 'r') as f:
        prompt_self_feedback = f.read()
    with open(args.path_to_revise_w_feedback, 'r') as f:
        prompt_revise_w_feedback = f.read()
    
    root_user_preference = ''
    if args.path_user_preference and os.path.exists(args.path_user_preference):
        with open(args.path_user_preference, 'r') as f:
            root_user_preference = f.read().strip()
    
    
    instructions, all_preferences, best_responses_1st, all_items = [], [], [], []
    with open(args.input, 'r') as f:
        content = f.readlines()
        for item in content:
            item = json.loads(item)
            if root_user_preference == 'preference_2':
                c_inst, best_response_1st = item['instruction'] + '\n\n' + item['preference_2'], item['best_response']
                all_preferences.append(item['preference_2'])
            elif root_user_preference == 'preference_1':
                c_inst, best_response_1st = item['instruction'] + '\n\n' + item['preference_1'], item['best_response']
                all_preferences.append(item['preference_1'])
            else:
                c_inst, best_response_1st = item['instruction'] + '\n\n' + root_user_preference, item['best_response']
                all_preferences.append(root_user_preference)

            
            instructions.append(c_inst.strip())
            best_responses_1st.append(best_response_1st)
            all_items.append(item)
    

    for i in tqdm(range(0, len(all_items), args.batch_size)):
        c_instructions, c_best_responses_1st, c_items, c_preferences = [], [], [], []
        for j in range(args.batch_size):
            if i + j <= len(all_items)-1:
                c_instructions.append(instructions[i+j])
                c_items.append(all_items[i+j])
                c_best_responses_1st.append(best_responses_1st[i+j])
                c_preferences.append(all_preferences[i+j])
        

        

        ## generate one feedback
        packed_1 = [prompt_self_feedback.format(question=question, answer = best_response_1st, preference = c_preference ) for question, best_response_1st, c_preference in zip(c_instructions, c_best_responses_1st, c_preferences)]
        
        c_issues_of_1st_response = evaluate_batch(args, model, prompter, packed_1, N=1)
        
        ## sample N responses
        packed_2 = [prompt_revise_w_feedback.format(question=question, answer=answer, preference=c_preference,  feedback=issue[0]) for question, answer, c_preference, issue in zip(c_instructions, c_best_responses_1st, c_preferences, c_issues_of_1st_response)]


        
        c_responses_2nd = evaluate_batch(args, model, prompter, packed_2, N=args.n_sample)


        for i, (response_1st, issue, response_2nd) in enumerate(zip(c_best_responses_1st, c_issues_of_1st_response, c_responses_2nd)):
            c_items[i]['response_1st'] = response_1st  
            c_items[i]['issue_of_1st_response'] = issue[0]
            c_items[i]['responses'] = response_2nd

        save_data_to_json(c_items, args.output)


    
def tree_search_wo_feedback(args):
    model = LLM(model=args.base_model, tensor_parallel_size=torch.cuda.device_count())
    prompter = Prompter(args.prompt_template)

    with open(args.path_to_revise_wo_feedback, 'r') as f:
        prompt_revise_wo_feedback = f.read()

    root_user_preference = ''
    if args.path_user_preference and os.path.exists(args.path_user_preference):
        with open(args.path_user_preference, 'r') as f:
            root_user_preference = f.read().strip()
    
    instructions, all_preferences, best_responses_1st, all_items = [], [], [], []
    with open(args.input, 'r') as f:
        content = f.readlines()
        for item in content:
            item = json.loads(item)
            if root_user_preference == 'preference_2':
                c_inst, best_response_1st = item['instruction'] + '\n\n' + item['preference_2'], item['best_response']
                all_preferences.append(item['preference_2'])
            elif root_user_preference == 'preference_1':
                c_inst, best_response_1st = item['instruction'] + '\n\n' + item['preference_1'], item['best_response']
                all_preferences.append(item['preference_1'])
            else:
                c_inst, best_response_1st = item['instruction'] + '\n\n' + root_user_preference, item['best_response']
                all_preferences.append(root_user_preference)
            
            instructions.append(c_inst.strip())
            best_responses_1st.append(best_response_1st)
            all_items.append(item)
    

    for i in tqdm(range(0, len(all_items), args.batch_size)):
        c_instructions, c_best_responses_1st, c_items, c_preferences = [], [], [], []
        for j in range(args.batch_size):
            if i + j <= len(all_items)-1:
                c_instructions.append(instructions[i+j])
                c_items.append(all_items[i+j])
                c_best_responses_1st.append(best_responses_1st[i+j])
                c_preferences.append(all_preferences[i+j])
        
        
        ## sample N responses w/o feedback
        packed_2 = [ prompt_revise_wo_feedback.format(question=question, answer = answer, preference = c_preference )  for question, answer, c_preference in zip(c_instructions, c_best_responses_1st, c_preferences)  ]

        
        c_responses_2nd = evaluate_batch(args, model, prompter, packed_2, N=args.n_sample)


        for i, (response_1st, response_2nd) in enumerate(zip(c_best_responses_1st, c_responses_2nd)):
            c_items[i]['response_1st'] = response_1st
            c_items[i]['responses'] = response_2nd

        save_data_to_json(c_items, args.output)

    


   
if __name__ == "__main__":
    main()

    

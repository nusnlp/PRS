from transformers import PreTrainedModel, LlamaConfig, LlamaModel, LlamaTokenizer
import torch.nn as nn
import torch
from typing import Optional, List
import jsonlines
import json
import numpy as np
import argparse
from tqdm import tqdm
import os

ultrarm_template= "Human: {instruction}\nAssistant: {response}"


def save_data_to_json(strings, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        writer = jsonlines.Writer(file)
        for string in strings:
            writer.write(string)

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


def main():
    parser = argparse.ArgumentParser(prog='', description='')
    parser.add_argument('-i,', '--input', required=True, help="folder to model outputs")
    parser.add_argument('--output', required=False, help="folder to model outputs")
    parser.add_argument('-m,', '--model', required=False, help="folder to model")
    parser.add_argument('-w,', '--weak', required=False, help="")
    parser.add_argument('--id_to_remove', default='path', required=False, help="")
    parser.add_argument('--path_user_preference', type=str, required=False, help="")
    


    args = parser.parse_args()

    root_user_preference = ''
    if args.path_user_preference and os.path.exists(args.path_user_preference):
        with open(args.path_user_preference, 'r') as f:
            root_user_preference = f.read().strip()
    


    ## load the model
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    model = LlamaRewardModel.from_pretrained(args.model)
    model.half()
    model.to('cuda')
    model.eval()

    def evaluate(instructions, responses):
        prompts = [ultrarm_template.format(instruction = instruction, response = response) for instruction, response in zip(instructions, responses)]
        print(prompts[0])
        print('\n\n')

        all_rewards = []
        #for prompt in tqdm(prompts):
        for prompt in prompts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
                reward = model(**inputs).item()
                all_rewards.append(reward)
            except Exception as e:
                print(f"{e}")
                all_rewards.append(0.0)

        return all_rewards


    max_rewards, mid_rewards = [], []

    data_to_save = []

    with open(args.input, 'r') as f:
        content = f.readlines()
        for item in tqdm(content):
            item = json.loads(item)
            
            if 'responses' in item:
                if root_user_preference == 'preference_2':
                    instruction = item['instruction'] + '\n\n' + item['preference_2']
                elif root_user_preference == 'preference_1':
                    instruction = item['instruction'] + '\n\n' + item['preference_1']
                else:
                    instruction = item['instruction']
                instruction = instruction.strip()


                responses = item[f'responses']
                instructions = [instruction] * len(responses)
                rewards = evaluate(instructions, responses)
                print(rewards)
                print('-------------------\n\n')
                max_rewards.append(max(rewards))
                mid_rewards.append(np.median(rewards))

                best_response = responses[np.argmax(rewards)]


                item['scores'] = rewards
                item['score'] = max(rewards)
                item['best_response'] = best_response


            else:
                if root_user_preference == 'preference_2':
                    instruction = item['instruction'] + '\n\n' + item['preference_2']
                elif root_user_preference == 'preference_1':
                    instruction = item['instruction'] + '\n\n' + item['preference_1']
                else:
                    instruction = item['instruction'] #+ '\n\n' + root_user_preference
                instruction = instruction.strip()



                response = item['response']
                rewards = evaluate([instruction], [response])
                print(rewards)
                print('-------------------\n\n')
                max_rewards.append(max(rewards))
                mid_rewards.append(np.median(rewards))
               
                item['scores'] = rewards
                item['score'] = max(rewards)


            data_to_save.append(item)


    print(f'Ave max rewards: {np.mean(max_rewards)}')    
    print(f'Ave median rewards: {np.mean(mid_rewards)}')


    save_data_to_json(data_to_save, args.output)





if __name__ == '__main__':
    main()      
        
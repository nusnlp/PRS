GPU=2

batch_size=10000
cal_reward=true

path_to_gen_self_feedback=./prompts/prompt_generate_self_feedback.v3.txt
path_to_revise_w_feedback=./prompts/prompt_revise_response_w_feedback.v3.txt
path_to_revise_wo_feedback=./prompts/prompt_revise_response_wo_feedback.txt

data_source=alpaca-GPT4-dev-100

prefer_type=common #preference_1  preference_2

if [ $data_source == 'alpaca-GPT4-dev-100' ]; then
    input=./data/exp_sampling/alpaca_gpt4.dev_set.num=100.w_preference_by_gpt-3.5.jsonl
fi

path_to_discriminator=./UltraRM-13b

for path_to_base_model in  vicuna-13b-v1.5  WizardLM-13B-V1.2  Mistral-7B-Instruct-v0.1 zephyr-7b-beta Xwin-LM-13B-V0.2  tulu-2-dpo-13b tulu-2-dpo-7b  ; do

root_to_save=./${data_source}.prefer=${prefer_type}.${path_to_base_model}
mkdir $root_to_save
max_tokens=2048

for mode in sample_N  sample_N_wo_prefer   tree_search tree_search_wo_feedback  ; do
for n_sample in   32 128 ; do

    if [ $path_to_base_model == 'vicuna-13b-v1.5' ]; then
        base_model=./models/vicuna-13b-v1.5
        prompt_template=./prompts/vicuna.json

    elif [ $path_to_base_model == 'WizardLM-13B-V1.2' ]; then
        base_model=./models/WizardLM-13B-V1.2
        prompt_template=./prompts/vicuna.json
    elif [ $path_to_base_model == 'Mistral-7B-Instruct-v0.2' ]; then
        base_model=./models/Mistral-7B-Instruct-v0.2
        prompt_template=./prompts/mistral-instruct.json

    elif [ $path_to_base_model == 'zephyr-7b-beta' ]; then
        base_model=./models/zephyr-7b-beta
        prompt_template=./prompts/zephyr.json
    
    elif [ $path_to_base_model == 'Xwin-LM-13B-V0.2' ]; then
        base_model=./models/Xwin-LM-13B-V0.2
        prompt_template=./prompts/vicuna.json
    
    elif [ $path_to_base_model == 'tulu-2-dpo-13b' ]; then
        base_model=./models/tulu-2-dpo-13b
        prompt_template=./prompts/tulu-2.json
    
    elif [ $path_to_base_model == 'tulu-2-dpo-7b' ]; then
        base_model=./models/tulu-2-dpo-7b
        prompt_template=./prompts/tulu-2.json
    
    elif [ $path_to_base_model == 'Mistral-7B-Instruct-v0.1' ]; then
        base_model=./models/llama2/Mistral-7B-Instruct-v0.1
        prompt_template=./prompts/mistral-instruct.json
    fi



if [ $mode == 'self_reflection' ]; then
    output=./$root_to_save/responses.mode=${mode}.prompt=v2.N=${n_sample}.jsonl
    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_self_reflection.py  --base_model $base_model --prompt_template $prompt_template --input $input --output $output --path_to_gen_self_feedback $path_to_gen_self_feedback --path_to_revise_w_feedback $path_to_revise_w_feedback --mode $mode --n_sample $n_sample --batch_size $batch_size --max_tokens $max_tokens

    if [ $cal_reward == true ]; then
        input=./$root_to_save/responses.mode=${mode}.prompt=v2.N=${n_sample}.jsonl
        output=./$root_to_save/reward.responses.mode=${mode}.prompt=v2.N=${n_sample}.jsonl
        log=./$root_to_save/out.responses.mode=${mode}.prompt=v2.N=${n_sample}
        CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_reward_from_ultraFeedback.py  --input $input --model $path_to_discriminator --output $output > $log
    fi


elif [ $mode == 'sample_N' ]; then
    if [ $prefer_type == 'common' ]; then
        path_user_preference=./user_preference.txt
    elif [ $prefer_type == 'preference_1' ]; then
        path_user_preference=./user_preference_1.txt
    elif [ $prefer_type == 'preference_2' ]; then
        path_user_preference=./user_preference_2.txt
    fi
    

    output=./$root_to_save/responses.mode=${mode}.N=${n_sample}.jsonl
    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_self_reflection.py  --base_model $base_model --prompt_template $prompt_template --input $input --output $output --path_to_gen_self_feedback $path_to_gen_self_feedback --path_to_revise_w_feedback $path_to_revise_w_feedback --mode $mode --n_sample $n_sample --path_user_preference $path_user_preference --batch_size $batch_size --max_tokens $max_tokens

    if [ $cal_reward == true ]; then
        input=./$root_to_save/responses.mode=${mode}.N=${n_sample}.jsonl
        output=./$root_to_save/reward.responses.mode=${mode}.N=${n_sample}.jsonl
        log=./$root_to_save/out.responses.mode=${mode}.N=${n_sample}
        CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_reward_from_ultraFeedback.py  --input $input --model $path_to_discriminator --output $output --path_user_preference $path_user_preference  > $log
    fi



elif [ $mode == 'sample_N_wo_prefer' ]; then
    output=./$root_to_save/responses.mode=${mode}.N=${n_sample}.jsonl
    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_self_reflection.py  --base_model $base_model --prompt_template $prompt_template --input $input --output $output --path_to_gen_self_feedback $path_to_gen_self_feedback --path_to_revise_w_feedback $path_to_revise_w_feedback --mode 'sample_N' --n_sample $n_sample  --batch_size $batch_size --max_tokens $max_tokens

    if [ $cal_reward == true ]; then
        input=./$root_to_save/responses.mode=${mode}.N=${n_sample}.jsonl
        output=./$root_to_save/reward.responses.mode=${mode}.N=${n_sample}.jsonl
        log=./$root_to_save/out.responses.mode=${mode}.N=${n_sample}
        CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_reward_from_ultraFeedback.py  --input $input --model $path_to_discriminator --output $output > $log
    fi

elif [ $mode == 'tree_search' ]; then
    n_sample=$(expr $n_sample / 2)
    ## phase 1: sample N / 2 responses
    if [ $prefer_type == 'common' ]; then
        path_user_preference=./user_preference.txt
    elif [ $prefer_type == 'preference_1' ]; then
        path_user_preference=./user_preference_1.txt
    elif [ $prefer_type == 'preference_2' ]; then
        path_user_preference=./user_preference_2.txt
    fi

    output=./$root_to_save/responses.mode=${mode}.stage=random_sampling.N=${n_sample}.jsonl
    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_self_reflection.py  --base_model $base_model --prompt_template $prompt_template --input $input --output $output --path_to_gen_self_feedback $path_to_gen_self_feedback --path_to_revise_w_feedback $path_to_revise_w_feedback --mode 'sample_N' --n_sample $n_sample --path_user_preference $path_user_preference --batch_size $batch_size --max_tokens $max_tokens

    ## phase 2: obtain the best response from phase 1
    input=./$root_to_save/responses.mode=${mode}.stage=random_sampling.N=${n_sample}.jsonl
    output=./$root_to_save/reward.responses.mode=${mode}.stage=random_sampling.N=${n_sample}.jsonl
    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_reward_from_ultraFeedback.py  --input $input --model $path_to_discriminator --output $output  --path_user_preference $path_user_preference

    ## phase 3: conduct self reflection
    input=./$root_to_save/reward.responses.mode=${mode}.stage=random_sampling.N=${n_sample}.jsonl
    output=./$root_to_save/responses.mode=${mode}.stage=self_reflection.N=${n_sample}.jsonl

    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_self_reflection.py  --base_model $base_model --prompt_template $prompt_template --input $input --output $output --path_to_gen_self_feedback $path_to_gen_self_feedback --path_to_revise_w_feedback $path_to_revise_w_feedback --mode 'tree_search' --n_sample $n_sample --path_user_preference $path_user_preference --batch_size $batch_size --max_tokens $max_tokens


    if [ $cal_reward == true ]; then
        input=./$root_to_save/responses.mode=${mode}.stage=self_reflection.N=${n_sample}.jsonl
        output=./$root_to_save/reward.responses.mode=${mode}.stage=self_reflection.N=${n_sample}.jsonl
        log=./$root_to_save/out.responses.mode=${mode}.stage=self_reflection.N=${n_sample}
        CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_reward_from_ultraFeedback.py  --input $input --model $path_to_discriminator --output $output --path_user_preference $path_user_preference > $log
    fi


elif [ $mode == 'tree_search_wo_feedback' ]; then
    n_sample=$(expr $n_sample / 2)
    ## phase 1: sample N / 2 responses
    if [ $prefer_type == 'common' ]; then
        path_user_preference=./user_preference.txt
    elif [ $prefer_type == 'preference_1' ]; then
        path_user_preference=./user_preference_1.txt
    elif [ $prefer_type == 'preference_2' ]; then
        path_user_preference=./user_preference_2.txt
    fi

    output=./$root_to_save/responses.mode=${mode}.stage=random_sampling.N=${n_sample}.jsonl
    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_self_reflection.py  --base_model $base_model --prompt_template $prompt_template --input $input --output $output --path_to_gen_self_feedback $path_to_gen_self_feedback --path_to_revise_w_feedback $path_to_revise_w_feedback --mode 'sample_N' --n_sample $n_sample --path_user_preference $path_user_preference --batch_size $batch_size --max_tokens $max_tokens


    ## phase 2: obtain the best response from phase 1
    input=./$root_to_save/responses.mode=${mode}.stage=random_sampling.N=${n_sample}.jsonl
    output=./$root_to_save/reward.responses.mode=${mode}.stage=random_sampling.N=${n_sample}.jsonl
    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_reward_from_ultraFeedback.py  --input $input --model $path_to_discriminator --output $output  --path_user_preference $path_user_preference

    ## phase 3: conduct self reflection
    input=./$root_to_save/reward.responses.mode=${mode}.stage=random_sampling.N=${n_sample}.jsonl
    output=./$root_to_save/responses.mode=${mode}.stage=self_reflection.N=${n_sample}.jsonl

    CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_self_reflection.py  --base_model $base_model --prompt_template $prompt_template --input $input --output $output --path_to_gen_self_feedback $path_to_gen_self_feedback --path_to_revise_w_feedback $path_to_revise_w_feedback --mode 'tree_search_wo_feedback' --n_sample $n_sample --path_user_preference $path_user_preference --batch_size $batch_size --max_tokens $max_tokens --path_to_revise_wo_feedback $path_to_revise_wo_feedback


    if [ $cal_reward == true ]; then
        input=./$root_to_save/responses.mode=${mode}.stage=self_reflection.N=${n_sample}.jsonl
        output=./$root_to_save/reward.responses.mode=${mode}.stage=self_reflection.N=${n_sample}.jsonl
        log=./$root_to_save/out.responses.mode=${mode}.stage=self_reflection.N=${n_sample}
        CUDA_VISIBLE_DEVICES=$GPU python ./code/generate_reward_from_ultraFeedback.py  --input $input --model $path_to_discriminator --output $output --path_user_preference $path_user_preference > $log
    fi
fi
done;
done;


done;


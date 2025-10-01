import os
import time
import argparse
import pandas as pd
from tqdm import tqdm

from const import crowd_enVent_emotions, get_prompt_pair, personas, experiencers


def get_experiment_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_hf",
        type=str,
        choices=['meta-llama/Meta-Llama-3.1-8B-Instruct',
                 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct',
                 'meta-llama/Meta-Llama-3.1-70B-Instruct',
                 'mistralai/Mistral-7B-Instruct-v0.3',
                 'Qwen/Qwen2-7B-Instruct']
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        help="used to track experiment meta information."
    )
    parser.add_argument(
        "--group_option",
        type=str,
        #choices=['ethnicity', 'nationality', 'religion']
        choices=['fan','none']
    )
    parser.add_argument(
        "--prompt_variation",
        type=str,
        choices=['origin',
                 'persona-1', 'persona-2', 'persona-3',
                 '1-person', '3-person',
                 '10-scale',
                 'no-persona'
                 ]
    )
    parser.add_argument(
        "--batch_id",
        type=int,
        default=0,
        help="to run experiments in parallel, for `ethnicity` there are batch 0-7; `nationality` with batch_id 0-9"
    )
    args = parser.parse_args()
    return args

def read_all_data(prompt_variation='origin'):
    df_list = []
    for emotion in crowd_enVent_emotions:
        df = pd.read_csv(
            'data/crowd-enVent_{}.tsv'.format(emotion), sep='\t')
        df_list.append(df)
    concat_df = pd.concat(df_list, sort=False)

    # Event reframe options
    if prompt_variation == '3-person':
        text_list = concat_df['third_person_text']
    elif prompt_variation == '1-person':
        text_list = concat_df['first_person_text']
    else:
        text_list = concat_df['generated_text']

    return {'id_list': concat_df['text_id'],
            'text_list': text_list,
            'emotion_list': concat_df['emotion']}


def prepare_prompts(data, prompt_variation, group_option,personas, experiencers):
    system_prompt, user_input = get_prompt_pair(prompt_variation)

    # idx, persona, experiencer, emotion, oid, text, system_prompt, user_input
    idx_list = []
    persona_list = []
    experiencer_list = []
    emotion_list = []
    oid_list = []
    text_list = []
    system_prompt_list = []
    user_input_list = []

    idx = 0
    
    if prompt_variation == "no-persona":
       
        for experiencer in experiencers:
            for emotion, text, oid in zip(data['emotion_list'], data['text_list'], data['id_list']):
                idx_list.append(idx)
                persona_list.append("None")   # persona 없음
                experiencer_list.append(experiencer)
                emotion_list.append(emotion)
                oid_list.append(oid)
                text_list.append(text)
                system_prompt_list.append(system_prompt)  # persona 포맷팅 필요 없음
                user_input_list.append(user_input.format(
                    experiencer=experiencer, sent=text, emotion=emotion))
                idx += 1
    else:
       
        for persona in personas:
            for experiencer in experiencers:
                for emotion, text, oid in zip(data['emotion_list'], data['text_list'], data['id_list']):
                    idx_list.append(idx)
                    persona_list.append(persona)
                    experiencer_list.append(experiencer)
                    emotion_list.append(emotion)
                    oid_list.append(oid)
                    text_list.append(text)
                    system_prompt_list.append(system_prompt.format(persona=persona))
                    user_input_list.append(user_input.format(
                        experiencer=experiencer, sent=text, emotion=emotion))
                    idx += 1

    prompt_folder = 'prompts'

    os.makedirs(prompt_folder, exist_ok=True)

    return pd.DataFrame({
        'idx': idx_list,
        'persona': persona_list,
        'experiencer': experiencer_list,
        'emotion': emotion_list,
        'oid': oid_list,
        'text': text_list,
        'system_prompt': system_prompt_list,
        'user_input': user_input_list
    }).to_csv("{}/{}_{}.tsv".format(prompt_folder, group_option, prompt_variation), sep='\t', index=False)


def vllm_inference(model_name, system_prompts, user_inputs, prompt_idx_list,persona_list,experinecer_list, emotion_list, exp_id, group_option, prompt_variation, bid, limit=6050*50):
    from vllm import LLM, SamplingParams

    # max_model_len = prompt length + output token length
    # maximum number of sequences per iteration: max_num_seqs=256 (used to avoid OOM error)
    #   => seems vLLM will set this automatically, same for max_num_batched_tokens=512
    # tensor_parallel_size: number of GPUs to use
    # Using more cards (e.g. 2) are faster, but couldn't promise greedy decoding
    if '70B' in model_name:
        model = LLM(
            seed=0,
            model=model_name,
            download_dir=os.environ['HF_HOME'],
            tensor_parallel_size=4,
            # gpu_memory_utilization=0.9,
            max_num_batched_tokens=512*10,
            max_model_len=512*10
        )
    else:
        model = LLM(
            seed=0,
            model=model_name,
            download_dir=os.environ['HF_HOME'],
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,
            max_model_len=2048,
            trust_remote_code=True
        )

    tokenizer = model.get_tokenizer()

    tot_len = len(system_prompts)
    max_limit_range = 15  # 14 * limit = 4235000
    limit_arr = [i*limit for i in range(max_limit_range) if i*limit < tot_len] + [tot_len]

    os.makedirs(exp_id, exist_ok=True)
    
    # for i, l in enumerate(limit_arr[1:]):
    # if os.path.isfile(path):
    #     continue

    print("Range {}: {}-{}".format(bid, limit_arr[bid], limit_arr[bid+1]))
    path = "{}/{}_{}_batch-{}.tsv".format(exp_id, group_option, prompt_variation, bid)

    system_prompts_set = system_prompts[limit_arr[bid]:limit_arr[bid+1]]
    user_inputs_set = user_inputs[limit_arr[bid]:limit_arr[bid+1]]

    # https://github.com/chujiezheng/chat_templates
    prompts = [tokenizer.apply_chat_template(
        [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input},
        ],
        tokenize=False,
        add_generation_prompt=True
    ) for system_prompt, user_input in zip(system_prompts_set, user_inputs_set)]

    outputs = model.generate(
        prompts,
        SamplingParams(
            max_tokens=128,
            temperature=0,
            stop_token_ids=[tokenizer.eos_token_id,
                            tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        )
    )

    idx_list = prompt_idx_list[limit_arr[bid]:limit_arr[bid+1]]

    return pd.DataFrame({
        'idx': idx_list,
        'persona' : persona_list[limit_arr[bid]:limit_arr[bid+1]],
        'experiencer' : experinecer_list[limit_arr[bid]:limit_arr[bid+1]],
        'system_prompt': system_prompts_set,
        'emotion': emotion_list[limit_arr[bid]:limit_arr[bid+1]],
        'user_input': user_inputs_set,
        'response': [outputs[idx].outputs[0].text for idx in range(len(idx_list))]
        
    }).to_csv(path, sep='\t', index=False)


def main(args):
    start_time = time.time()
    print('{}_{}'.format(args.group_option, args.prompt_variation))

    # Read data and prepare prompts
    data = read_all_data(args.prompt_variation)
    prepare_prompts(data, args.prompt_variation, args.group_option,personas, experiencers)

    # Read prompts and model inference
    prompt_folder = 'prompts'
    read_prompts = pd.read_csv(
        "{}/{}_{}.tsv".format(prompt_folder, args.group_option, args.prompt_variation), sep='\t')
    vllm_inference(args.model_name_hf, read_prompts['system_prompt'], read_prompts['user_input'],
                   read_prompts['idx'], read_prompts['persona'], read_prompts['experiencer'], read_prompts['emotion'], 
                   args.exp_id, args.group_option, args.prompt_variation, args.batch_id)

    elapsed_time = time.time() - start_time
    print("[Used time] {:.4f} minutes".format(elapsed_time / 60))


if __name__ == '__main__':
    main(get_experiment_configs())

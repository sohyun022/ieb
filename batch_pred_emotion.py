import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.utils import resample
import pandas as pd
import math
from google.colab import files
from vllm import LLM, SamplingParams
from const import crowd_enVent_emotions, group_mappings, get_prompt_pair


def get_experiment_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str,
                        choices=[
                            'meta-llama/Llama-3.1-8B-Instruct',
                            'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct',
                            'meta-llama/Meta-Llama-3.1-70B-Instruct',
                            'mistralai/Mistral-7B-Instruct-v0.3',
                            'Qwen/Qwen2-7B-Instruct'
                        ])
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--group_option", type=str, choices=['KBO_fan','MLB_fan'])
    parser.add_argument("--prompt_variation", type=str,
                        choices=['origin', 'persona-1', 'persona-2', 'persona-3',
                                 '1-person', '3-person', '10-scale', 'no-persona'])
    parser.add_argument("--batch_id", type=int, default=0)
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


def prepare_prompts(data, prompt_variation, group_option):

    persona_group = group_mappings[group_option] + ['a person']
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
    
    for persona in tqdm(persona_group):
        for experiencer in persona_group:
            for emotion, text, oid in zip(data['emotion_list'], data['text_list'], data['id_list']):
                idx_list.append(idx)
                persona_list.append(persona)
                experiencer_list.append(experiencer)
                emotion_list.append(emotion)
                oid_list.append(oid)
                text_list.append(text)
                system_prompt_list.append(
                    system_prompt.format(persona=persona))
                user_input_list.append(user_input.format(
                    experiencer=experiencer, sent=text, emotion=emotion))
                idx += 1

    prompt_folder = 'prompts'
    os.makedirs(prompt_folder, exist_ok=True)

    df = pd.DataFrame({
        'idx': idx_list,
        'perceiver': persona_list,
        'experiencer': experiencer_list,
        'emotion': emotion_list,
        'oid': oid_list,
        'text': text_list,
        'system_prompt': system_prompt_list,
        'user_input': user_input_list
    })
    
    save_path = f"{prompt_folder}/{group_option}_{prompt_variation}.tsv"
    df.to_csv(save_path, sep='\t', index=False)
    print(f"Prompts saved at: {save_path}")

    return df

def vllm_inference(model_name, system_prompts, user_inputs, prompt_idx_list,
                   perceiver_list, experiencer_list, emotion_list,
                   exp_id, group_option, prompt_variation, bid, limit=6050*50):

    # 모델 로드
    if '70B' in model_name:
        model = LLM(
            seed=0,
            model=model_name,
            download_dir=os.environ['HF_HOME'],
            tensor_parallel_size=4,
            max_num_batched_tokens=512*10,
            max_model_len=512*10
        )
    else:
        model = LLM(
            seed=0,
            model=model_name,
            download_dir=os.environ['HF_HOME'],
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1,
            max_model_len=1024,
            trust_remote_code=True
        )

    tokenizer = model.get_tokenizer()
    tot_len = len(system_prompts)

    # 전체 데이터를 limit 단위로 자동 분할
    num_batches = math.ceil(tot_len / limit)
    limit_arr = [i * limit for i in range(num_batches)] + [tot_len]

    num_batches = math.ceil(tot_len / limit)

    print(f"총 프롬프트 개수: {tot_len}")
    print(f"배치 크기 (limit): {limit}")
    print(f"총 배치 수: {num_batches}") 

    # batch id 검증
    if bid >= len(limit_arr) - 1:
        print(f"⚠️ Batch ID {bid} is out of range (max batch id: {len(limit_arr) - 2})")
        return

    start, end = limit_arr[bid], limit_arr[bid + 1]
    print(f"Range {bid}: {start}-{end} (총 {end - start}개, 전체 {tot_len}개 중)")


    # 결과 경로 준비
    content_path = "/content"
    exp_path = os.path.join(content_path, exp_id)
    os.makedirs(exp_path, exist_ok=True)
    path = os.path.join(exp_path, f"{group_option}_{prompt_variation}_batch-{bid}.tsv")

    # 현재 배치 데이터 슬라이싱
    system_prompts_set = system_prompts[start:end]
    user_inputs_set = user_inputs[start:end]
    idx_list = prompt_idx_list[start:end]

    # apply_chat_template
    prompts = [
        tokenizer.apply_chat_template(
            [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': usr_input},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        for sys_prompt, usr_input in zip(system_prompts_set, user_inputs_set)
    ]

    # 모델 inference
    outputs = model.generate(
        prompts,
        SamplingParams(
            max_tokens=128,
            temperature=0,
            stop_token_ids=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        )
    )

    # outputs 인덱싱 안전하게
    responses = []
    for i, o in enumerate(outputs):
        try:
            responses.append(o.outputs[0].text)
        except Exception as e:
            print(f"⚠️ Warning: output {i} 처리 중 오류 - {e}")
            responses.append("")

    # 결과 저장
    df = pd.DataFrame({
        'idx': idx_list,
        'perceiver': perceiver_list[start:end],
        'experiencer': experiencer_list[start:end],
        'system_prompt': system_prompts_set,
        'emotion': emotion_list[start:end],
        'user_input': user_inputs_set,
        'response': responses
    })

    df.to_csv(path, sep='\t', index=False)
    print(f"Saved: {path} ({len(df)} rows)")

    return df

def main(args):
    start_time = time.time()
    print('{}_{}'.format(args.group_option, args.prompt_variation))

    # Read data and prepare prompts
    data = read_all_data(args.prompt_variation)
    # Read prompts and model inference
    prompt_folder = 'prompts'
    read_prompts = prepare_prompts(data, args.prompt_variation, args.group_option)

    vllm_inference(
        args.model_name_hf,
        read_prompts['system_prompt'],
        read_prompts['user_input'],
        read_prompts['idx'],
        read_prompts['perceiver'],
        read_prompts['experiencer'],
        read_prompts['emotion'],
        args.exp_id,
        args.group_option,
        args.prompt_variation,
        args.batch_id
    )

    elapsed_time = time.time() - start_time
    print("[Used time] {:.4f} minutes".format(elapsed_time / 60))


if __name__ == '__main__':
    main(get_experiment_configs())

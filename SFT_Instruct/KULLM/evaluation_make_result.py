import pandas as pd
import openai

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from peft import PeftModel
from utils.prompter import Prompter

import argparse
import os
import json
import time

parser=argparse.ArgumentParser()
parser.add_argument("--peft_model", type=str, default=None)
parser.add_argument("--openai_key", type=str, default="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")
parser.add_argument("--base_model", type=str, default="nlpai-lab/kullm-polyglot-12.8b-v2")
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--output_file", type=str, default='./output/out.csv')
parser.add_argument("--gpt", type=bool, default=True)

args=parser.parse_args()

# openai.api_key = "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"
openai.api_key=args.openai_key
MODEL = args.base_model
prompter = Prompter("kullm")


def infer_from_gpt(instruction, input) :
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": 'user',
             'content': prompter.generate_prompt(instruction, input)
            },
        ],
        temperature = 0.5)
    return response['choices'][0]['message']['content']



finetuned=True

#
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)

model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

def infer_from_original(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)
    return result


# 데이터 불러오기
prompts=[]
dataset_type='prompt_only'
with open(f'./{args.dataset}', 'r', encoding='utf-8') as f:
    if os.path.splitext(args.dataset)[1]=='.jsonl':
        for line in f:
            json_data = json.loads(line)
            if dataset_type=='prompt_only':
                prompts.append(json_data['instruction'])
            else:
                for inst in json_data['instances']:
                    prompts.append({'instruction': json_data['instruction'], 'input' : inst['input'] })
    else:
        prompts = f.readlines()

COLUMNS = ['instruction', 'input', 'prompt', 'model_output' ,'base_model_output', 'gpt3_output']
df = pd.DataFrame(columns=COLUMNS)
count = 0
for prompt in prompts:
    if dataset_type=='prompt_only':
        instruction=prompt
        input_text=None
    else:
        instruction=prompt['instruction']
        input_text=prompt['input']
    backoff_time = 10
    retry_cnt=0
    row = {'instruction':instruction, 'input': input_text, 'prompt': prompter.generate_prompt(instruction=instruction, input=input_text)}
    if args.gpt:
        while retry_cnt<10:
            try:
                output = infer_from_gpt(instruction, input_text)
            except openai.error.OpenAIError as e:
                if "Please reduce your prompt" in str(e):
                    target_length = int(target_length * 0.8)
                    print(f"Reducing target length to {target_length}, retrying...")
                else:
                    print(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 1.5
                    retry_cnt+=1
                output=str(e)
        row['gpt3_output']=output
    try:
        output = infer_from_original( instruction=instruction, input_text=input_text)
    except Exception as e:
        output=str(e)
    row['base_model_output']=output
    # score = extract_scores_from_string(make_evaluation(instruction, output))
    # print(f"instruction : {instruction}")
    # print(f"output : {output}")
    # print(f"evaluation: {score}\n")
    df.append(row,ignore_index=True)
    count += 1


def infer_from_fintuned(
    instruction,
    input=None,
    temperature=0.2,
    num_beams=5,
    max_new_tokens=512,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device=f"cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        num_beams=num_beams,
        **kwargs,
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield prompter.get_response(output)


if finetuned:
    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
        torch_dtype=torch.float16,
    ).to(device=f"cuda", non_blocking=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    idx=0
    for prompt in prompts:
        if dataset_type=='prompt_only':
            instruction=prompt
            input_text=None
        else:
            instruction=prompt['instruction']
            input_text=prompt['input']
        try:
            output = infer_from_fintuned(instruction=instruction, input=input_text)
        except Exception as e:
            output=str(e)
        df.loc[idx,'model_output']=output
        idx+=1


df.to_csv(f"./{args.output_file}", encoding='utf-8')
    
"""
{'role':'user','content': '평가 기준:\
    - 친근함 (1 - 5): Response가 친근한 답변을 제공했나요?\
    - 이해 가능성 (1 - 5): Instruction에 기반하여 Response를 이해할 수 있나요?\
    - 자연스러움 (1 - 5): Instruction을 고려했을 때 자연스러운 Response인가요?\
    - 맥락 유지 (1 - 5): Instruction을 고려했을 때 Response가 맥락을 유지하나요?\
    - 전반적인 품질 (1 - 5): 위의 답변을 바탕으로 이 발언의 전반적인 품질에 대한 인상은 어떤가요?'
},
"""
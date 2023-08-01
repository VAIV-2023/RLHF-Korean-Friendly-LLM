import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from peft import PeftModel
from utils.prompter import Prompter
import time

MODEL = "nlpai-lab/kullm-polyglot-12.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model = PeftModel.from_pretrained(
    model,
    "./lora_weights/final+",
    torch_dtype=torch.float16,
).to(device=f"cuda", non_blocking=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)
prompter = Prompter("kullm")

model.eval()

def infer_from_original(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=1.0, num_beams=1, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result

def infer_from_fintuned(
    instruction,
    input=None,
    temperature=1.0,
    num_beams=1,
    max_new_tokens=512,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device=f"cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        num_beams=num_beams,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
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


# 데이터 불러오기
with open('./data/prompts/'+'all_prompt.txt', 'r', encoding='utf-8') as f:
    prompts = f.readlines()

times=[]

for prompt in prompts:
    instruction = prompt  
    start = time.time() 
    output = infer_from_fintuned(instruction=instruction)
    end = time.time()
    times.append(end-start)
    result=""
    for s in output:
        result+=s
    result = result.split("endoftext​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​")[0]
    result=result.strip()
    print(result)

print(sum(times)/len(times))
            

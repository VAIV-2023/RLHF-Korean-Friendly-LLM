import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.prompter import Prompter

MODEL = "nlpai-lab/kullm-polyglot-5.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

prompter = Prompter("kullm")

def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result


result = infer(input_text="안녕, 반가워")
print(result)
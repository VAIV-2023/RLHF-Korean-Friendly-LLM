import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from peft import PeftModel
from utils.prompter import Prompter

MODEL = "nlpai-lab/kullm-polyglot-12.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model = PeftModel.from_pretrained(
    model,
    "conversation",
    torch_dtype=torch.float16,
).to(device=f"cuda", non_blocking=True)
model.eval()

prompter = Prompter("kullm")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

def infer_from_original(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result

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


result = infer_from_original(input_text="나 어제 여행을 다녀왔어")
print(result)

output = infer_from_fintuned(instruction="나 어제 여행을 다녀왔어")
result=""
for s in output:
    result+=s
result = result.split("<|endoftext|>​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​")[0]
print(result)

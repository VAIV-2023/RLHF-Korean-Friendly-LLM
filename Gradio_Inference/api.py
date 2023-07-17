import os
import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, GPTNeoXForCausalLM, GPTNeoXTokenizerFast

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

from flask import Flask, Response, request
from flask_ngrok import run_with_ngrok
from flask_cors import CORS
import json

app = Flask(__name__)
run_with_ngrok(app)
CORS(app)

@app.route('/')
def index():
    return 'CARE_BOT'

device = "cuda"
load_8bit: bool = False
base_model: str = "nlpai-lab/kullm-polyglot-12.8b-v2"
lora_weights: str = "kullm_lora_weight"
prompt_template: str = "kullm"

prompter = Prompter(prompt_template)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(base_model)

def load():
    model = GPTNeoXForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
      
    return model

def evaluate(
    instruction,
    model,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
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

@app.route('/api/main', methods=['POST'])
def main():
    response = Response()
    if request.method == 'POST':
        response.headers.add("Access-Control-Allow-Origin", "*")
        instruction = request.get_json()
        responses = evaluate(instruction)
        result = ""
        for response in responses:
            result += response
        result = result.split("<|endoftext|>")[0]
        result = result.split(instruction)[1]
        result = result.replace('\n','')
        response.set_data(json.dumps(result))
        return response

if __name__ == "__main__":
    model = load()
    app.run()
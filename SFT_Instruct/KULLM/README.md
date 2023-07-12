<p align="center" width="100%">
<img src="assets/logo.png" alt="NLP Logo" style="width: 90%;">
</p>

## Update Logs
- 2023.06.23: [한국어 대화 평가 결과 공개](https://github.com/nlpai-lab/KULLM#evaluation)
- 2023.06.08: [🤗Polyglot-ko 5.8B 기반 KULLM-Polyglot-5.8B-v2 fp16 모델 공개](https://huggingface.co/nlpai-lab/kullm-polyglot-5.8b-v2)
- 2023.06.01: [구름(KULLM) 데이터셋 v2](https://huggingface.co/datasets/nlpai-lab/kullm-v2) HuggingFace Datasets 공개
- 2023.05.31: [🤗Polyglot-ko 12.8B 기반 KULLM-Polyglot-12.8B-v2 fp16 모델 공개](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2)
- 2023.05.30: [🤗Polyglot-ko 12.8B 기반 KULLM-Polyglot-12.8B fp16 모델](https://huggingface.co/metterian/kullm-polyglot-12.8b) 공개

---

<br>

# ☁️ KULLM (구름): Korea University Large Language Model

KULLM(구름)은 고려대학교 [NLP & AI 연구실](http://blp.korea.ac.kr/)과 [HIAI 연구소](http://hiai.korea.ac.kr)가 개발한 한국어 Large Language Model (LLM) 입니다.

구름 프로젝트는 한국어 모델 뿐만 아니라, 데이터 셋까지 전면 공개하여 한국어 LLM 생태계에 기여하고자 하였습니다.

<br/>

## Example

<img src="assets/example.png" width="65%" >

<br/>

## Backbone Model?: Polyglot-ko

KULLM(구름)은 Backbone Model로 [Polyglot-ko](https://github.com/EleutherAI/polyglot)을 사용하여 학습을 진행하였습니다.

1. Polyglot-ko 5.8B 기반-v2 -> 🤗 [nlpai-lab/kullm-polyglot-5.8b-v2](https://huggingface.co/nlpai-lab/kullm-polyglot-5.8b-v2)
    - 데이터셋 v2: [GPT4ALL](https://github.com/nomic-ai/gpt4all), [Dolly](https://github.com/databrickslabs/dolly), [Vicuna](https://github.com/lm-sys/FastChat)
2. Polyglot-ko 12.8B 기반-v2 -> 🤗 [nlpai-lab/kullm-polyglot-12.8b-v2](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2)
    - 데이터셋 v2: [GPT4ALL](https://github.com/nomic-ai/gpt4all), [Dolly](https://github.com/databrickslabs/dolly), [Vicuna](https://github.com/lm-sys/FastChat)
3. Polyglot-ko 12.8B 기반-v1 -> 🤗 [metterian/kullm-polyglot-12.8b-v1](https://huggingface.co/metterian/kullm-polyglot-12.8b-v1)
    - 데이터셋 v1: GPT4ALL

Meta의 LLaMA 모델을 백본으로 만든 모델은 테스트 결과 한국어 성능이 좋지 못하여 공개하지 않기로 했습니다. 추후 여러 좋은 한국어 성능을 보여주는 LLM 모델을 학습하여 공개하고자 합니다.

<br/>

## KULLM 모델 실행 예시 코드

### Huggingface Pipeline으로 실행

- 최신버전 torch / HF 라이브러리 설치

```bash
pip install -U torch transformers tokenizers accelerate
```

아래 예제 코드로 실행해볼 수 있습니다.

```python
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


result = infer(input_text="고려대학교에 대해서 알려줘")
print(result)
# '고려대학교에 대해 궁금한 점이 있으시면 언제든지 문의해 주세요. 고려대학교는 한국에서 가장 오래되고 권위 있는 대학교 중 하나로, 고려대학교의 역사는 한국의 역사와 함께해 왔습니다. 고려대학교는 학문적 우수성을 추구하는 동시에 사회적 책임을 다하기 위해 최선을 다하고 있습니다. 고려대학교는 학생, 교수진, 교직원을 위한 다양한 프로그램과 지원을 제공하는 것으로 유명합니다. 고려대학교는 한국의 정치, 경제, 사회 분야에서 중요한 역할을 담당하고 있습니다. 고려대학교에 대해 더 자세히 알고 싶으신가요?'
```

<br/>

## Dataset

### 구름 데이터셋 v2

[HuggingFace Datasets](https://huggingface.co/datasets/nlpai-lab/kullm-v2)

구름 데이터셋 v2는 [GPT4ALL](https://github.com/nomic-ai/gpt4all), [Vicuna](https://github.com/lm-sys/FastChat), 그리고 Databricks의 [Dolly](https://github.com/databrickslabs/dolly) 데이터셋을 병합한 것입니다. 이 모든 데이터셋은 DeepL을 이용하여 한국어로 번역되었습니다.

GPT4ALL은 instruction tuned assistant-style language model이며, Vicuna와 Dolly 데이터셋은 다양한 자연어 처리 문제를 해결하는 데 활용됩니다. 특히, Dolly는 instruction/response fine tuning records를 훈련 데이터로 사용한 언어 모델입니다.

```python
from datasets import load_dataset

ds = load_dataset("nlpai-lab/kullm-v2", split="train")
ds
DatasetDict({
    train: Dataset({
        features: ['id', 'instruction', 'input', 'output'],
        num_rows: 152630
    })
})
```

### 구름 데이터셋 v1

[구름 데이터셋 v1](./data/kullm-v1.jsonl)은 GPT4ALL을 기반으로 합니다.

#### 데이터셋 예시

GPT4ALL 데이터셋은 다음과 같이 Instruct 부분과 Input, 그리고 Output 부분으로 구성되어있습니다.

```json
{
    "id": "user_oriented_task_235",
    "motivation_app": "Yelp",
    "instruction": "전문 분야에 따라 레스토랑, 홈 서비스, 자동차 서비스, 기타 중 하나로 비즈니스를 분류합니다.",
    "instances": [
        {
            "input": "견적을 받으려면 650-636-4884로 전화하거나 웹사이트를 방문하세요. 이 매장은 신품 타이어 및 일반 자동차 수리를 전문으로 합니다. 모든 타이어를 자체적으로 보유하고 있으며 예산이나 차량 특성에 맞는 다양한 타이어를 보유하고 있습니다. 어떤 타이어가 필요한지 잘 모르시겠다면 전문가가 상주하여 고객의 요구에 가장 적합한 타이어를 선택할 수 있도록 도와드립니다. 또한 상용차 타이어도 취급하고 있어 다양한 차량에 맞는 타이어를 제공할 수 있습니다.",
            "output": "Auto Services"
        }
    ]
},
```

한국어로 번역된 데이터셋은 [`kullm-v2.jsonl`](data/kullm-v2.jsonl)에 저장되어 있습니다.

<br>

## Training with LoRA

KULLM은 한국어 모델로 Polyglot 12.8B 모델을 Low Rank Adaptation (LoRA)를 사용하여 학습하였습니다.

모델 학습은 A100 80GB 4대로 진행했습니다. 학습에 사용한 코드는 [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)을 기반으로 사용하였습니다.

### KULLM v2

🤗 Huggingface Repo: [https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2)

모델 학습은 구름 데이터셋 v2 (GPT4ALL, Dolly, Vicuna)을 사용하여 진행했습니다. 총 8 epoch 학습하였으며, A100 80GB 4대를 사용했습니다.

### KULLM v1

🤗 Huggingface Repo: 🤗 [https://huggingface.co/metterian/kullm-polyglot-12.8b-v1](https://huggingface.co/metterian/kullm-polyglot-12.8b-v1)

모델 학습은 구름 데이터셋 v1 (GPT4ALL)을 사용하여 진행했습니다. 총 5 epoch 학습하였으며, A100 80GB 4대를 사용했습니다.

### Dependency

1. 다음 명령어를 통해 필요한 패키지를 설치:

```bash
pip install -r requirements.txt
```

2. 만약 bitsandbytes가 작동하지 않는다면, [소스에서 직접 설치하세요](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md). 윈도우 사용자는 [다음의 설명서](https://github.com/tloen/alpaca-lora/issues/17)를 참조하세요.

### Traning (`finetune_polyglot.py`)

이 코드는 Polyglot 모델에 Parameter-Efficient Fine-Tuning (PEFT)을 적용하고, 프롬프트 구성 및 토크나이징에 관련된 코드가 들어있는 파일입니다.

사용 예시:

```
python finetune_polyglot.py \
--base_model='EleutherAI/polyglot-ko-12.8b' \
--data_path='./data/kullm-v2.jsonl'
```

다음과 같이 하이퍼파라미터를 조정 가능합니다:

```bash
python -m torch.distributed.launch  --master_port=34322  --nproc_per_node 4 finetune_polyglot.py \
    --fp16 \
    --base_model 'EleutherAI/polyglot-ko-12.8b' \
    --data_path data/kullm-v2.jsonl \
    --output_dir ckpt/$SAVE_DIR \
    --prompt_template_name kullm \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "[query_key_value, xxx]" \
    --train_on_inputs \
    --logging_steps 1 \
    --eval_steps 40 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --group_by_length
```

<br/>

## Evaluation

- 대화 평가 메트릭 (Dialogue Evaluation Metric)을 사용하여 모델 간 한국어 대화를 평가 했습니다. 대화 평가 메트릭은 G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Yang Liu. et. al. 2023)과 USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation (Shikib Mehri. et. al. 2020)을 활용하여 평가 Prompt를 구성했습니다. 
- 평가 모델은 GPT-4를 사용하였고, 평가 데이터셋은 [yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)의 휴먼 평가 데이터셋인 `user_oriented_instructions.jsonl`을 deepl로 번역한 데이터셋을 사용하였습니다.
- 해당 데이터셋은 [`user_oriented_instructions_eval.jsonl`](data/user_oriented_instructions_eval.jsonl)에 저장되어 있습니다.


![eval_result](./assets/eval.png)
- 그래프의 값은 0-100점으로 스케일링 되었습니다. 
### LLM Inference Results for Korean Evaluation Set

| Type   | Base-model     | Model                                                        | 이해 가능성 (0 - 1) | 자연스러움 (1 - 3) | 맥락 유지 (1 - 3) | 흥미롭기 (1 - 3) | 지시어 사용 (0-1) | 전반적인 품질  (1-5) |
| ------ | ----------------- | ------------------------------------------------------------ | :---------------: | :----------------: | :---------------: | :--------------: | :---------------: | :------------------: |
| Closed | GPT3.5-turbo      | GPT-3.5                                                      | 0.980              | 2.806              | 2.849             | 2.056            | 0.917             | 3.905                |
| Closed | GPT-4             | GPT-4                                                        | 0.984             | 2.897              | 2.944             | 2.143            | 0.968             | 4.083                |
|  |  |  |  |  |  |  |  |  |
| Open   | Polyglot-ko-12.8b | [KoAlpaca v1.1](https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B) | 0.651             | 1.909              | 1.901             | 1.583            | 0.385             | 2.575                |
| Open   | LLaMA-7b          | [koVicuna](https://huggingface.co/junelee/ko_vicuna_7b)      | 0.460              | 1.583              | 1.726             | 1.528            | 0.409             | 2.440                 |
| Open   | Polyglot-ko-12.8b | [ KULLM v2](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2) | **0.742**         | **2.083**          | **2.107**         | **1.794**        | **0.548**         | **3.036**            |


### Prompt

```
두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction), 입력(Input)을 받게 될 것입니다. 그리고 지시문과 입력에 대한 응답(Response)이 제시됩니다.
당신의 작업은 응답을 평가 단계에 따라 응답을 평가하는 것입니다.
이 평가 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다. 평가하는 동안 이 문서를 계속 열어두고 필요할 때 참조해 주세요.

평가 기준:
- 이해 가능성 (0 - 1): Input에 기반하여 Response를 이해 할 수 있나요?
- 자연스러움 (1 - 3): 사람이 자연스럽게 말할 법한 Instruction 인가요?
- 맥락 유지 (1 - 3): Input을 고려했을 때 Response가 맥락을 유지하나요?
- 흥미롭기 (1 - 3): Response가 지루한가요, 아니면 흥미로운가요?
- Instruction 사용 (0 - 1): Instruction에 기반하여 Response를 생성 했나요?
- 전반적인 품질 (1 - 5): 위의 답변을 바탕으로 이 발언의 전반적인 품질에 대한 인상은 어떤가요?

평가 단계:
1. Instruction, Input, 그리고 Response을 주의깊게 읽습니다.
2. 위의 평가 기준에 따라 Response을 평가합니다.

Instruction:
{{instruction}}

Input:
{{input}}

Response:
{{response}}


Result
- 이해 가능성 (0 - 1):
- 자연스러움 (1 - 3):
- 맥락 유지 (1 - 3):
- 흥미롭기 (1 - 3):
- Instruction 사용 (0 - 1):
- 전반적인 품질 (1 - 5):
```

<br/>

---

## Citation

Please cite the repo if you use the data or code in this repo.

```
@misc{kullm,
  author = {NLP & AI Lab and Human-Inspired AI research},
  title = {KULLM: Korea University Large Language Model Project},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nlpai-lab/kullm}},
}
```

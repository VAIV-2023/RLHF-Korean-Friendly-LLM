VAIV2023
## GPT 기반의 자연스럽고 윤리적인 일상 대화형 챗봇 모델

### 과제 목표
GPT3 기반 자연스럽고 윤리적인 한국어 기반 일상 대화형 챗봇 모델 구현

### 개발 내용
한국어 LLM 학습 데이터셋 구축

SFT 제작

Reward Model 학습

RLHF 강화 학습

초거대 언어 모델을 고속, 저비용으로 학습할 수 있는 DeepSpeed Chat 이용

### Task1. LLM 학습 데이터셋 구축
일상대화 데이터셋: 국립국어원 모두의 말뭉치 일상 대화 데이터셋

혐오표현 데이터셋: AIHUB 텍스트 윤리 검증 데이터셋

RLHF 번역 데이터셋: DeepSpeedChat에서 공개한 Reward Model 학습 데이터셋

Evol-Instruct 데이터셋: 다양한 분야에 대한 복잡하고 논리적인 prompt와 답변 존재

Self-Instruct 데이터셋: 사람이 직접 생성한 양질의 Seed 데이터를 기반으로 GPT-3.5를 이용해 데이터 증강

KoBEST 데이터셋: 평가용 데이터셋, Common Sense & Inference 능력과 관련된 HellaSwag 데이터셋

### Task2. SFT 모델 제작
Pretrained Model: KULLM 12.8B

Evaluation: G-Eval 방식

일상 대화 능력과 혐오 표현 대처 능력 각각에 대하여 평가

### Task3. Reward Model 학습
Base Model: Polyglot-ko 1.3B

모델 응답 결과들을 비교평가하여 사람의 선호도 데이터를 학습

SFT 모델과 GPT-3.5로부터 prompt당 2개의 Response 생성 후 선호도 평가


2023 성균관대 하계집중 산학협력프로젝트 VAIV
# GPT 기반의 자연스럽고(Friendly) 윤리적인(Harmless) 일상 대화형 챗봇 모델

# 과제 목표
    GPT3 기반 자연스럽고 윤리적인 한국어 기반 일상 대화형 챗봇 모델 구현
- Self-Instruct: GPT4를 이용한 데이터 증강
- RLHF(Reinforcement Learning from Human Feedback): 사람의 선호도를 반영한 강화학습
- DeepSpeed: 대규모 분산 딥러닝을 위한 새로운 메모리 최적화 기술
  
### 개발 내용
    한국어 LLM 학습 데이터셋 구축

    SFT 제작

    Reward Model 학습

    RLHF 강화 학습

    초거대 언어 모델을 고속, 저비용으로 학습할 수 있는 DeepSpeed Chat 이용

### Task1. LLM 학습 데이터셋 구축
    ![단계별 데이터셋](https://github.com/VAIV-2023/VAIV2023/assets/79634774/fd9cb0a2-58a3-43aa-8116-b420680641fd)

    일상대화 데이터셋: 국립국어원 모두의 말뭉치 일상 대화 데이터셋

    혐오표현 데이터셋: AIHUB 텍스트 윤리 검증 데이터셋

    RLHF 번역 데이터셋: DeepSpeedChat에서 공개한 Reward Model 학습 데이터셋

    Evol-Instruct 데이터셋: 다양한 분야에 대한 복잡하고 논리적인 prompt와 답변 존재

    Self-Instruct 데이터셋: 사람이 직접 생성한 양질의 Seed 데이터를 기반으로 GPT-3.5를 이용해 데이터 증강

    KoBEST 데이터셋: 평가용 데이터셋, Common Sense & Inference 능력과 관련된 HellaSwag 데이터셋

### Task2. SFT 모델 제작
    Pretrained Model: KULLM 12.8B

    일상대화 데이터셋과 혐오표현 데이터셋에 대한 Instruction Set으로 Fine-Tuning 진행

    Evaluation: G-Eval 방식

    일상 대화 능력과 혐오 표현 대처 능력 각각에 대하여 평가

### Task3. Reward Model 학습
    Base Model: Polyglot-ko 1.3B

    모델 응답 결과들을 비교평가하여 사람의 선호도 데이터를 학습

    SFT 모델과 GPT-3.5로부터 prompt당 2개의 Response 생성 후 선호도 평가

### Task4. RLHF 강화학습
    1) Trainable SFT가 prompt에 대한 response 생성
     
    2) Reward Model이 Response에 대한 Reward값 r 도출
     
    3) Trainable SFT와 Reference SFT에서의 response 토큰별 확률 분포 차이를 패널티로 적용해 최종 reward를 계산
     
    4) PPO 알고리즘으로 Trainable SFT를 업데이트




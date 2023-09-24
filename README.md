2023 성균관대 하계집중 산학협력프로젝트 VAIV
## GPT 기반의 자연스럽고(Friendly) 윤리적인(Harmless) 일상 대화형 챗봇 모델

# 과제 목표
    GPT-NEOX 기반 자연스럽고 윤리적인 한국어 기반 일상 대화형 챗봇 모델 구현
- Self-Instruct: GPT4를 이용한 데이터 증강
- RLHF(Reinforcement Learning from Human Feedback): 사람의 선호도를 반영한 강화학습
- DeepSpeed: 대규모 분산 딥러닝을 위한 새로운 메모리 최적화 기술
  
# 개발 내용
    Task 1: 강화학습 단계별 데이터셋 구축
    Task 2: SFT 모델 Fine-tuning
    Task 3: Reward 모델 구현
    Task 4: RLHF와 DeepSpeedChat을 통한 최종 모델 구현
    Task 5: 최종 모델 성능 평가

# Task1. LLM 학습 데이터셋 구축
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/a4988abd-c6fd-4fc2-8e53-9a02240e2275)
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/dae49a1e-a834-463c-9f95-34cf254fdaeb)
## 데이터셋 선정 시 고려 사항
- **일상 대화와 혐오 표현 대처 능력을 올리기 위한 데이터셋과, 학습 시 챗봇 모델의 general한 task에 대한 성능이 하락하는 것을 막기 위해서 general task 데이터셋을 구성**
  
- **국립국어원 일상 대화 데이터셋:** 일상적인 대화에 대한 자연스러운 응답이 있으면서도, 맞춤법이 잘 지켜지고 은어, 비문, 초성 등이 없으며 주제별로 다양한 대화가 있음
  
- **AI Hub 혐오 표현 데이터셋:** 혐오, 차별, 성적인 내용, 폭력, 범죄 등 카테고리별로 다양한 혐오 표현이 있음
  
- **General task 데이터셋**
    - Evol-Instruct 데이터셋: 다양한 분야에 대한 복잡하고 논리적인 prompt와 답변이 있음
    - Self-Instruct 데이터셋: 사람이 직접 생성한 양질의 Seed data를 기반으로 데이터 증강
    - RLHF 한국어 번역 데이터셋: DeepSpeedChat에서 공개한 데이터셋을 한국어로 번역



 



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




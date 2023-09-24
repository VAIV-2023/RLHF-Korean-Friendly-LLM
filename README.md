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
    Task 3: Reward 모델 ver1,2,3 구현
    Task 4: RLHF와 DeepSpeedChat을 통한 최종 모델 구현

# Task1. 강화학습 단계별 데이터셋 구축
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

# Task2. SFT 모델 Fine-tuning
## Baseline Model
[- 고려대학교 NLP & AI 연구실과 HIAI 연구소가 개발한 한국어 LLM **"KULLM"** 사용](https://github.com/nlpai-lab/KULLM)

## Datasets
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/085610db-3714-43c3-855b-58baad2f4e8b)

## SFT Model Finetuning 
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/0f5e36fa-20a8-43f9-bd03-5f8224d5e9d0)
* 모델학습에는 Google Colab에서 제공하는 A100 40GB GPU 사용
  
## SFT Model Evaluation
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/9fe9e5aa-6dc7-4c7b-8529-45e0a75db9c6)
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/a994a960-db7c-4e75-a11a-d7755d372722)
* G-Eval: https://arxiv.org/abs/2303.16634

## Final SFT Model
- https://huggingface.co/Trofish/KULLM-SFT-v2

# Task3-1. Reward Model ver1 구현
## Baseline Model
- EleutherAI에서 개발한 초거대 한국어 언어 모델 **Polyglot-Ko** 사용
- 1.3b 모델과 5.8b 모델을 각각 실험
## Datasets
![image](https://github.com/VAIV-2023/RLHF-Korean-Friendly-LLM/assets/79634774/0082da9b-b0b8-4089-8647-cffa5ce724fb)
- InstructGPT의 데이터셋 구축 방법
    - Reward 모델 학습 데이터셋으로 SFT 학습에 사용한 prompt(1,500개 - 일상대화:혐오표현=2:1)와 새로운 prompt(1,000개 - DeepSpeedChat 번역 데이터셋) 사용 
    - SFT 모델에서 한개의 prompt당 K개의 Response를 생성하고, 순위를 Labeling
- 데이터셋 라벨링
    - Instruct GPT의 경우 사람이 직접 Labeling을 하엿지만, 일관된 평가와 시간 단축을 위해 GPt-4와 G-Eval을 이용
    - SFT에서 생성한 두 Response 중 G-Eval 평가 점수 합이 높은 것을 Chosen response로 결정
    - 데이터셋 유형별로 G-Eval 평가 Prompt에 차이를 두었음
    -   ![image](https://github.com/VAIV-2023/RLHF-Korean-Friendly-LLM/assets/79634774/7d7117d0-02e9-42dd-8ce3-5244cf726bf8)
## Reward v1 Model Finetuning
- ![image](https://github.com/VAIV-2023/RLHF-Korean-Friendly-LLM/assets/79634774/da4d9b15-ec91-44bb-84d9-f28aeffd16ad)
- InstructGPT 논문에 따르면, Reward 모델은 overfitting되면 성능이 크게 저하된다고 함 --> epoch 수를 1로 설정
- batch size나 learning rate 등 다른 hyper-parameter는 성능에 큰 영향이 없다고 함
- Colab A100 40GB 기준 총 학습 시간 4분

## Reward v1 Model Evaluation
- ![image](https://github.com/VAIV-2023/RLHF-Korean-Friendly-LLM/assets/79634774/f4af0b7d-af47-4881-8adf-d14be43c0eb1)
- Reward Model Template
  - **"아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요. \n\n ### 명령어:\n{prompt}\n\n ### 응답:\n"**

# Task3-2. Reward Model ver2,3 구현
- ![image](https://github.com/VAIV-2023/RLHF-Korean-Friendly-LLM/assets/79634774/f4af0b7d-af47-4881-8adf-d14be43c0eb1)

# Task4. RLHF와 DeepSpeedChat을 통한 최종 모델 구현
- Microsoft에서 만든 대규모 분산 딥러닝을 위한 새로운 메모리 최적화 기술(DeepSpeed)을 RLHF Process에 적용한 DeepSpeedChat 사용
- Human preference로 학습을 시킨 Reward 모델과 강화학습을 통해 SFT 모델에 사람의 선호도를 반영하여 자연스럽고(FRIENDLY), 윤리적인 (HARMLESS) 챗봇 생성
  
## Baseline Models
- Actor Model: KULLM-SFT-V2
- Reward Model: Polyglot-Ko-Reward-V3

## Training Options
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/ae2cdfe5-7552-4009-a99a-244e79d945dc)

## RLHF Training
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/3d4dbf68-5222-4f6a-a6d0-87ea176c5211)
- 학습 결과, SFT 모델의 답변에 대한 퀄리티인 Reward가 상승하는 것을 확인 (사람의 선호도가 높은 답변을 생성)

## RLFH Model Evaluation
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/2b58ed3a-7ed5-4e60-ba4b-c9b291b1fdff)
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/75b2a1ee-d7c0-4ba9-ab2f-727abab644e9)

## Final RLHF Model
- https://huggingface.co/Trofish/KULLM-RLHF



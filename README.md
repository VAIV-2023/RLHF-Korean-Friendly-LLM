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

# Task2. SFT 모델 제작
## Baseline Model
[- 고려대학교 NLP & AI 연구실과 HIAI 연구소가 개발한 한국어 LLM **"KULLM"** 사용](https://github.com/nlpai-lab/KULLM)

## Datasets
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/085610db-3714-43c3-855b-58baad2f4e8b)

## SFT Finetuning 
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/0f5e36fa-20a8-43f9-bd03-5f8224d5e9d0)
* 모델학습에는 Google Colab에서 제공하는 A100 40GB GPU 사용
  
## SFT Evaluation
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/9fe9e5aa-6dc7-4c7b-8529-45e0a75db9c6)
![image](https://github.com/VAIV-2023/VAIV2023/assets/79634774/a994a960-db7c-4e75-a11a-d7755d372722)
* G-Eval: https://arxiv.org/abs/2303.16634

## Final Model
- https://huggingface.co/Trofish/KULLM-SFT-v2

# Task3. Reward Model 학습



# Task4. RLHF 강화학습





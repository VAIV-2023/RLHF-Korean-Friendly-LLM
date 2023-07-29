## 1. Step 2 데이터셋

`data_final/` 폴더가 최종 데이터셋
|파일 이름|설명|데이터 개수|
|---|---|---|
|step2_KULLM_train.json|최종 학습 데이터셋<br>(conversation_train.json + hatespeech_train.json + translation_train.json)|2484|
|step2_KULLM_test.json|최종 테스트 데이터셋<br>(conversation_test.json + hatespeech_test.json + translation_test.json)|100|
|koAlpaca_train.json|바이브팀에서 받은 데이터(KoAlpaca로 뽑은 response 라벨링 데이터)|2524|
|koAlpaca_train.json|바이브팀에서 받은 데이터에서 랜덤으로 test set을 구성|100|
|step2_KULLM_KoAlpaca_train.json|step2_KULLM_train.json + koAlpaca_train.json| 6008|
|step2_KULLM_KoAlpaca_test.json|step2_KULLM_test.json + step2_KULLM_KoAlpaca_test.json|100|

**데이터셋 예시**

```
{
    "id2": "koAlpaca2528",
    "prompt": "크리스마스 식사 준비에 대해 어떤 조언을 해주실 수 있나요?",
    "chosen": "크리스마스 식사를 준비할 때는 그 분위기와 모임성격을 고려해야 합니다. 예를 들어, 간단한 스낵과 음료를 함께 즐기는 파티라면 핑거 푸드를 준비하고 각자가 먹고 싶은 음식을 싸오도록 하면 좋습니다. 또, 한 상에 모여 앉아 이야기를 나누며 천천히 식사를 즐기는 가족모임이라면, 샐러드, 전채요리, 파스타, 육류, 과일 등의 순서로 서빙하면 좋습니다.",
    "rejected": "크리스마스 식사를 준비할 때는 몇 가지 팁을 참고하면 좋습니다. 1) 메뉴 선택이 중요합니다. 자신이 있는 요리나 좋아하는 음식 중에서 골라보세요. 2) 미리 계획하고 재료를 손질해두면 좋습니다. 3) 가족 모임이니만큼, 많은 요리를 만들어 함께 나눠먹어보세요."
}
```

데이터마다 id를 부여함  
step1과 step2에는 겹치는 데이터가 있어 id(step1 id), id2(step2 id)를 따로 주었음

## 2. 라벨링 코드

### 1) G-eval

- **코드:**

```
/labeling
  ├── conversation-geval.py
  ├── labeling/hatespeech-geval.py
  └── labeling/translation-geval.py
```

- **installation:**

```
pip install openai
```

- 일상대화, 혐오표현, 번역 데이터셋마다 평가 항목이 다름
- 일상대화, 혐오표현은 평가 항목 6개, 번역 데이터셋은 '진실성' 항목 추가하여 7개
- G-eval을 하면 response마다 score 합계가 나옴

### 2) ranking

- **코드: `labeling/rank.py`**
- 두 response의 score 합계를 비교해 chosen/reject 형식으로 파일 변환
- 혐오표현과 일상대화는 두 답변 score 모두 20(or 19) 미만인 경우 아예 제외함
- 두 답변의 score가 같을 경우 제외함
- 혐오표현의 경우 제외된 답변들 중 일부는 직접 검수하여 train set에 넣기도 함

## 3. Reward Model 학습 방법

- **shell script 코드:** `run_350m.sh`
- **script의 주요 argument 설명 :**
  |Argument|설명|
  |--|---|
  |--model_name_or_path|HuggingFace 모델 그대로 넣으면 됨|
  |--data_path|`local/json`으로 하고, train.json과 eval.json 파일을 DeepSpeedChat/data에 넣기 |
  |--data_split |`0,1,0`: 넣은 데이터를 step2에서만 사용한다는 뜻 |
  |--num_padding_at_beginning|`0` 우리 모델은 처음 padding이 없으므로 0|

### 코드 실행 방법

**준비사항:**  
DeepSpeedExamples clone 받기
가상 환경 세팅 및 requirements 설치

```
cd DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning
bash training_scripts/single_gpu/run_350m.sh
```

- log는 step2_reward_model_finetuning/output 폴더 저장됨
- 모델도 같은 폴더에 저장됨

## 4. Reward Model 평가 방법

평가 코드: `rw_eval.py`
실행 예시:

```
python rw_eval.py  --model_name_or_path ./1.3b-KULLM_template --data_path ./data/hatespeech_test.py
```

- `DeepSpeed-Chat/training/step2_reward_model_finetuning` 폴더에 rw_eval.py 코드가 있는데, 기존 코드는 response에 대한 점수만 알려주어서 accuracy를 계산하는 코드를 추가하였음
- Accuracy = chosen answer의 score를 더 높게 생성한 개수/전체 prompt 개수

## 5. 학습된 Reward Model

- prompt template 적용한 버전: 슈퍼컴퓨팅센터 .folder/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/1.3b-KULLM_template
- prompt template 적용하기 전: .folder/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/1.3b-KULLM_prev

- train set: step2_KULLM_train.json
- test set: step2_KULLM_test.json

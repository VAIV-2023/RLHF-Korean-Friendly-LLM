import pandas as pd
import json
import openai
from tqdm import tqdm
import jsonlines

openai.api_key = "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"

# id 정보를 모아서 저장하는 예시
data_list= []
prompt_format = {
    "instruction" : "test instruction",
    "input" : "",
    "output" : "test output"
    } 

def make_response(instruction) :
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "system", "content": "두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction)을 받게 될 것입니다.\
                                                당신의 작업은 지시문과 입력에 해당하는 답변(response)을 생성하는 것입니다.\
                                                너무 길지 않고 간결하게 답변을 생성해주세요."},
                    {'role':'user','content': f'\
                     Instruction: {instruction}\
                     Response: '},
                    ],
        temperature = 0.5)
    return response['choices'][0]['message']['content']


def make_evaluation(instruction, output) :
    response = openai.ChatCompletion.create(
        model = "gpt-4",
        messages = [{"role": "system", "content": "두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction)을 받게 될 것입니다.\
                                                그리고 지시문과 입력에 대한 응답(Response)이 제시됩니다.\
                                                당신의 작업은 응답(Response)을 평가 단계에 따라 응답을 평가하는 것입니다.\
                                                이 평가 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다. 평가하는 동안 이 문서를 계속 열어두고 필요할 때 참조해 주세요."},
                    {'role':'user','content': '평가 기준:\
                     - 공감 (1 - 5): Response가 Instruction에 충분히 공감하여 답변을 제공 했나요?\
                     - 친근함 (1 - 5): Response가 친구가 말한 것 처럼 친근한 답변을 제공 했나요?\
                     - 자연스러움 (1 - 5): 친구가 자연스럽게 말할 법한 Instruction 인가요?\
                     - 흥미롭기 (1 - 5): Response가 지루한가요, 아니면 흥미로운가요?\
                     - 전반적인 품질 (1 - 5): 위의 답변을 바탕으로 이 발언의 전반적인 품질에 대한 인상은 어떤가요?'},                   
                    {'role':'user','content': f'평가 단계:\
                     1. Instruction, Input, 그리고 Response을 주의깊게 읽습니다.\
                     2. 위의 평가 기준에 따라 Response을 평가합니다.\
                     Instruction: {instruction}\
                     Response:{output}'},
                    {'role':'system','content': f'Result:\n - 공감 (1 - 5): \n - 친근함 (1 - 5): \n - 자연스러움 (1 - 5): \n - 흥미롭기 (1 - 5): \n - 전반적인 품질 (1 - 5): \n\n'}
                    ],
        temperature = 0.5)
    return response['choices'][0]['message']['content']


# GPT-4의 답변에서 점수만 뽑아쓰기
def extract_scores_from_string(text):
    scores = []
    lines = text.split("\n")
    for line in lines:
        if "-" in line and ":" in line:
            score_str = line.split(":")[-1].strip()
            try:
                score = float(score_str)
            except:
                score = 0
            scores.append(score)
    return scores

COLUMNS = ['Empathy','friendliness','Naturalness','Interesting','quality']
df = pd.DataFrame(columns=COLUMNS)

# 데이터 불러오기
emotion_conversations = pd.read_excel('./emotion_conversation.xlsx', index_col = 0) # 옵션: 인덱스 칼럼 제외

count = 0
for instruction in emotion_conversations['사람문장1']:
    print(instruction)
    prompt_format['instruction'] = instruction    
    if count > 100: break # 100개만 돌리고 끝
    try:        
        prompt_format['output'] = make_response(instruction)        
        score = extract_scores_from_string(make_evaluation(prompt_format['instruction'],prompt_format['output']))
        print(f"instruction : {prompt_format['instruction']}")
        print(f"output : {prompt_format['output']}")
        print(f"evaluation: {score}\n\n")
        newDF = list()
        newDF = score
        newDF = pd.DataFrame(data=[newDF], columns = COLUMNS) 
        df = pd.concat([df,newDF])       
        df.to_csv("./chatgpt_evaluation_emotion_conversation.csv")  
        count += 1
    except:
        print("error occur!")
        continue
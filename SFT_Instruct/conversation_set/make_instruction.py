#{"id": "user_oriented_task_0", "motivation_app": "Grammarly", "instruction": "주어진 문장이 너무 장황하거나 복잡하거나 불분명할 수 있습니다. 문장을 다시 작성하고 간결하게 유지하여 글을 더 명확하게 만드세요. 가능하면 복잡한 문장은 여러 문장으로 나누고 불필요한 단어는 제거하세요.", "instances": [{"input": "제 요금에 대해 궁금한 점이 있거나 이 프로젝트의 범위를 늘리거나 줄여야 할 필요가 있는 경우 알려주세요.", "output": "If you have any questions about my rate or find it necessary to increase or decrease this project's scope, please let me know."}]}

import json
import random

with open('./selected.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

line=[]
id=0
for category in data:
    li = data[category]
    for item in li:
        tmp={}
        tmp['id'] = "conversaion_"+str(id)
        id +=1
        tmp['instruction'] = item['input']
        tmp['output'] = item['output']
        line.append(tmp)

with open('./instructions.jsonl', 'w') as f:
    for item in line:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + '\n')

import json

with open('./step2_final_dataset.json', 'r') as f:
    data = json.load(f)

"""with open('./data_final/conversation_train.json', 'r') as f:
    data = json.load(f)"""

li=[]
for item in data:
    if 'conversation' in item['id']:
        if len(item['rejected'])<8: li.append(item)
        
with open('./tmp.json', 'w')as f:
    json.dump(li, f, ensure_ascii=False, indent=4)
import json
import random

with open('./instructions.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
    
existing=[]
for item in data:
    existing.append(item['instruction'])

with open('./dialogues.json', 'r') as f:
    data = json.load(f)

cnt=0
id=1514
li=[]
while(cnt<37):
    for key in data:
        rand = random.randint(0, len(data[key]))
        inst = data[key][rand]['input']
        output = data[key][rand]['output']
        if len(inst)<15: pass
        elif len(output)<15: pass
        else:
            if (inst not in li) and (inst not in existing): 
                cnt+=1
                li.append({
                    "id": "conversation_"+str(id),
                    "instruction": inst,
                    "output": output
                })
                id+=1

with open('./tmp.jsonl', 'w') as f:
    for item in li:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')
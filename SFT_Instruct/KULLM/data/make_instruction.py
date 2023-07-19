import json
import random

conversation = []
hatespeech = []
evol = []

with open('conversation.jsonl', 'r') as f:
    id=0
    for line in f:
        data = json.loads(line)
        tmp={}
        tmp['id']='conversation_'+str(id)
        id+=1
        tmp['instruction'] = data['instruction']
        tmp['input']=""
        tmp['output']= data['output']
        conversation.append(tmp)

with open('hatespeech.jsonl', 'r') as f:
    id=0
    for line in f:
        data = json.loads(line)
        tmp={}
        tmp['id']='hatespeech'+str(id)
        id+=1
        tmp['instruction'] = data['instruction']
        tmp['input']=""
        tmp['output']= data['output']
        hatespeech.append(tmp)

with open('evol.jsonl', 'r') as f:
    id=0
    for line in f:
        data = json.loads(line)
        tmp={}
        tmp['id']='evol'+str(id)
        id+=1
        tmp['instruction'] = data['instruction']
        tmp['input']=data['input']
        tmp['output']= data['output']
        evol.append(tmp)

all= conversation + hatespeech + evol
random.shuffle(all)

with open('./instructions_all.jsonl', 'w') as f:
    for item in all:
        json_line = json.dumps(item, ensure_ascii=False)
        f.write(json_line + '\n')

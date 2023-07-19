import random

conversation = []
hatespeech=[]

with open('./conversation_prompt.txt', 'r') as f:
    data = f.readlines()
    for item in data:
        conversation.append(item)

with open('./hatespeech_prompt.txt', 'r') as f:
    data = f.readlines()
    for item in data:
        hatespeech.append(item)

all=[]

selected_conversation = random.sample(conversation, 160)
selected_hatespeech = random.sample(hatespeech, 40)

all.extend(selected_conversation)
all.extend(selected_hatespeech)

random.shuffle(all)
with open('./all_prompt.txt', 'w') as f:
    for item in all:
        f.write(item)
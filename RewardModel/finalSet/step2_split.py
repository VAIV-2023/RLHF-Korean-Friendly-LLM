import json
import random as rd

conv=[]
hate=[]
evol=[]

with open('step2_train_final.json') as f:
    data=json.load(f)
    for d in data:
        if 'conversation' in d['id']:
            conv.append(d)
        elif 'hatespeech' in d['id']:
            hate.append(d)
        elif 'evol_inst' in d['id']:
            evol.append(d)

    rd.shuffle(conv)
    rd.shuffle(hate)
    rd.shuffle(evol)

    conv_train= conv[50:]
    conv_test= conv[:50]
    hate_train= hate[50:]
    hate_test= hate[:50]
    evol_train= evol[50:]
    evol_test= evol[:50]

    conv_train.sort(key = lambda x : int(x["id"].replace('conversation_','')))
    conv_test.sort(key = lambda x : int(x["id"].replace('conversation_','')))
    hate_train.sort(key = lambda x : int(x["id"].replace('hatespeech','')))
    hate_test.sort(key = lambda x : int(x["id"].replace('hatespeech','')))
    evol_train.sort(key = lambda x : int(x["id"].replace('evol_instruction_','')))
    evol_test.sort(key = lambda x : int(x["id"].replace('evol_instruction_','')))

    train=conv_train+hate_train+evol_train
    test=[conv_test,hate_test,evol_test]
    f=['conv','hate','general']

with open('step2_train.json','w') as a:
    a.write(json.dumps(train,ensure_ascii=False,indent=4))

for i in range(3):
    with open(f'step2_test_{f[i]}.json','w') as b:
        b.write(json.dumps(test[i],ensure_ascii=False,indent=4))
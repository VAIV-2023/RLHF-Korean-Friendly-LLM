import re
import os
import json
with open('out_conv_final.txt') as r, open('conversation_prompt.txt') as j:
    prompts=r.readlines()
    selected_prompts=j.readlines()
    # selected=json.load(j)
    # selected_prompts=[i["input"]+'\n' for i in selected]
    prompts=[i.strip() for i in prompts]
    prompts_dict={}
    selected_dict={}
    for p in prompts:
        prompts_dict[p.replace(' ','')]=p
    selected_prompts=[i.strip() for i in selected_prompts]
    for p in selected_prompts:
        selected_dict[p.replace(' ','')]=p
    print(prompts)
    print(selected_prompts)
    rem_dup = list(set(prompts_dict.keys()) - set(selected_dict.keys()))
    w=open('final_out.txt','w')
    for i in rem_dup:
        w.write(prompts_dict[i]+'\n')
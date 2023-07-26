import pandas as pd

file='./conversation/gpt_conversation_eval.csv'

df = pd.read_csv(file)
df = df.iloc[:, 1:]  # Remove the first column

df.to_csv(file, index=False, encoding='utf-8')
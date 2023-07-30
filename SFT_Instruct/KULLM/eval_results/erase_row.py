import pandas as pd

file='./hatespeech/kullm_ftv3_hatespeech_eval.csv'

df = pd.read_csv(file)
df = df.iloc[:, 1:]  # Remove the first column

df.to_csv(file, index=False, encoding='utf-8')
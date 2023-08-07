import pandas as pd

model='custom'
task='conversation'

file='./'+task+'/'+model+'_'+task+'_eval.csv'

df = pd.read_csv(file)
df = df.iloc[:, 1:]  # Remove the first column

df.to_csv(file, index=False, encoding='utf-8')
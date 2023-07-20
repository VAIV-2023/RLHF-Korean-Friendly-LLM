import pandas as pd

# CSV 파일을 읽어서 DataFrame 객체로 변환
data = pd.read_csv('./all/kullm_orig_all_eval.csv')

# 원하는 열 선택
columns = ['친근함', '무해함', '이해 가능성', '자연스러움', '맥락 유지', '전반적인 품질']
selected_data = data[columns]

# 열의 평균값 계산
avg = selected_data.mean()

# 각 열의 평균값 출력
print(avg)
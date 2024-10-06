import pandas as pd

# CSV 파일 읽기
file1 = pd.read_csv('./food_review1.csv')
file2 = pd.read_csv('./food_review2.csv')

# 파일 1의 첫 번째 110행 + 파일 2의 나머지 행(111행부터 끝까지)
merged_file = pd.concat([file1.iloc[:110], file2[111:]], ignore_index=True)

# 결과를 새로운 CSV 파일로 저장
merged_file.to_csv('./food_review.csv', index=False)

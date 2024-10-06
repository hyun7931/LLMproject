import re
import pandas as pd

def preprocess_text(text):
    # 이모지 및 비ASCII 문자 제거 (이모지만 제거하고 나머지 유지)
    text = re.sub(r'[^\w\s,.!?]', '', text)  # \w는 단어 문자, \s는 공백 문자
    # \n을 공백으로 변경
    text = text.replace('\n', ' ')
    return text


def preprocess_text_csv(file, type):
    csv_path = f'./{file}.csv'
    filepath = f"./{file}_pre.csv"
    df = pd.read_csv(csv_path)

    df_review = df["reviews"]
    df_col = f"{type}_name"

    for i in range(len(df_review)):
        preprocessed_list = []

        print("*", i+1, "번째 식당: ", df[df_col].iloc[i])

        for j in range(len(eval(df_review.iloc[i]))):  # eval로 리스트 변환
            preprocessed_data = preprocess_text(eval(df_review.iloc[i])[j])
            preprocessed_list.append(preprocessed_data)

            print(j+1, "번째 리뷰:")
            print(preprocessed_data)
        df.at[i, "reviews"] = preprocessed_list
        print("="*50)

    print()
    print(df)

    df.to_csv(filepath, index=False, encoding='utf-8-sig')

preprocess_text_csv('hotel_review', 'hotel')
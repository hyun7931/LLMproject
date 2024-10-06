import pandas as pd
import sqlite3

# CSV 파일 읽기
file2 = './cafe_sreview_keyword_cosine.csv'

# SQLite 데이터베이스 연결
conn = sqlite3.connect('my_database.db')

# 'restaurants' 테이블 생성
create_table_query = '''
CREATE TABLE IF NOT EXISTS restaurants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    keyword TEXT NOT NULL
);
'''
# 테이블 생성 쿼리 실행
conn.execute(create_table_query)

# CSV 파일에서 데이터 읽기
data = pd.read_csv(file2)

# 'restaurants' 테이블에 데이터 추가
for index, row in data.iterrows():
    name = row['cafe_name']
    keyword = row['top_keywords']
    type_ = 'cafe'  # 고정된 값

    # SQL INSERT 문 실행
    conn.execute("INSERT INTO restaurants (name, type, keyword) VALUES (?, ?, ?)", (name, type_, keyword))

# 변경 사항 저장
conn.commit()

# 데이터베이스에서 데이터 출력
query = "SELECT * FROM restaurants"
output_data = pd.read_sql_query(query, conn)

# 출력
print(output_data)

# 연결 종료
conn.close()

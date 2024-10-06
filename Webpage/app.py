from flask import Flask, render_template, request, jsonify, redirect, url_for
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


app = Flask(__name__)

def search_db(query):
    conn = sqlite3.connect('my_database.db')  # 실제 데이터베이스 경로로 변경
    cursor = conn.cursor()

    # 검색어와 일치하는 name과 type을 가져옴
    # 검색어와 정확히 일치하는 name과 해당하는 type을 가져옴
    cursor.execute("SELECT name, type FROM restaurants WHERE name = ?", (query,))
    results = cursor.fetchall()
    
    conn.close()
    return results

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir="./models")
model = BertModel.from_pretrained(model_name, cache_dir="./models")

def embed_text(text):
    # 텍스트를 토크나이즈하고 텐서로 변환
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # BERT 모델을 통해 임베딩 계산
    with torch.no_grad():
        outputs = model(**inputs)

    # 임베딩 추출 (여기서는 마지막 은닉 상태의 평균을 사용)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def get_embedding(keywords):
    # 키워드 리스트에 대한 임베딩 생성
    keyword_embeddings = [embed_text(keyword) for keyword in keywords]
    
    # 키워드 임베딩을 스택하여 하나의 텐서로 변환
    return torch.vstack(keyword_embeddings)

def connect_db():
    conn = sqlite3.connect('my_database.db')
    return conn

@app.route('/calculate-cosine-similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()
    places = data['places']
    
    # DB 연결
    conn = connect_db()
    cursor = conn.cursor()
    
    # 유사도 결과를 저장할 리스트
    similarity_results = []
    
    for place in places:
        # 선택된 place의 name에 해당하는 keyword 가져오기
        cursor.execute("SELECT keyword FROM restaurants WHERE name=?", (place['name'],))
        target_keyword = cursor.fetchone()
        
        if target_keyword:
            target_embedding = get_embedding(target_keyword[0])
            
            # DB에서 모든 name, type, keyword 가져오기 (자기 자신 제외)
            cursor.execute("SELECT name, type, keyword FROM restaurants WHERE name != ?", (place['name'],))
            all_data = cursor.fetchall()
            
            # 각 keyword에 대한 임베딩 계산 및 코사인 유사도 계산
            all_embeddings = [get_embedding(row[2]) for row in all_data]  # 모든 keyword 임베딩
            similarities = cosine_similarity([target_embedding], all_embeddings)[0]
            
            # 유사도가 높은 상위 5개 데이터 추출
            top_5_indices = np.argsort(-similarities)[:5]
            top_5_results = [(all_data[idx][0], all_data[idx][1], all_data[idx][2]) for idx in top_5_indices]
            
            similarity_results.append(top_5_results)
    
    # DB 연결 종료
    conn.close()

    return jsonify(similarity_results)


@app.route('/select-place/search', methods=['POST'])
def search():
    query = request.form.get('query')  # 검색어를 가져옴
    results = search_db(query)  # 데이터베이스에서 검색
    
    # 검색 결과를 JSON으로 반환
    return jsonify(results)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/select-place')
def selection1():
    return render_template('selection1.html')

@app.route('/select-place/recommend')
def selection2():
    return render_template('selection2.html')

@app.route('/plan')
def plan():
    return render_template('plan.html')

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000,debug=True)

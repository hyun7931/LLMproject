import sqlite3
from langchain_text_splitters import CharacterTextSplitter
import google.generativeai as genai
import numpy as np
import faiss
import requests
from bs4 import BeautifulSoup
import os
import schedule
import time
import re
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_upstage import ChatUpstage
import os

# Google API 키 설정
GOOGLE_API_KEY = "AIzaSyCoQCVuQqVzTgW6EjbM93_72A-HFz4AcCk"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

api_key = "up_fdSahrxOU6kfvdVqP5eN0SY5R87Uk"
#api_key=os.getenv("up_fdSahrxOU6kfvdVqP5eN0SY5R87Uk")
llm = ChatUpstage(api_key=api_key, model="solar-1-mini-chat")

placedb_path = r"C:\practiceLLM1\JJtotal777.db"
timedb_path = r"C:\practiceLLM1\JJtraveltimes.db"
eventdb_path = r"C:\practiceLLM1\events.db"
newsdb_path = r"C:\practiceLLM1\news.db"

# 데이터베이스 연결 및 데이터 가져오기 함수
def fetch_data(db_path, query):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    return data

def placedoc(placedb_path, destinations):
    conn = sqlite3.connect(placedb_path)
    cursor = conn.cursor()
    
    # destinations 리스트를 쿼리의 WHERE 절에 포함하기 위한 조건 생성
    placeholders = ', '.join('?' for _ in destinations)  # ? 플레이스홀더 생성
    query = f"""
        SELECT name, classify, category 
        FROM newtotals 
        WHERE name IN ({placeholders})
    """
    
    # 데이터 가져오기
    cursor.execute(query, destinations)
    result = cursor.fetchall()
    conn.close()
    
    # 결과를 thisplacelist에 저장
    thisplacelist = []
    for row in result:
        name, classify, category = row
        thisplacelist.append({
            "name": name,
            "classify": classify,
            "category": category
        })

    #print("thisplacelist를 출력합니다.")
    #print(thisplacelist)
    return thisplacelist

# 최대 문서 번호 가져오기
def get_max_num(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f'SELECT MAX(num) FROM {table_name}')
    max_num = cursor.fetchone()[0]
    conn.close()
    return max_num if max_num is not None else 0

# 링크에서 내용 크롤링
def fetch_content(link, content_div_class):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')
    content_div = soup.find('div', class_=content_div_class)
    return content_div.get_text(strip=True) if content_div else ''

# 이벤트 데이터베이스 생성
def create_eventdb(db_path):
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            num INTEGER UNIQUE,
            title TEXT,
            link TEXT,
            기간 TEXT,
            시간 TEXT,
            연령대 TEXT,
            입장료 TEXT,
            장소 TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 뉴스 데이터베이스 생성
def create_newsdb(db_path):
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            num INTEGER UNIQUE,
            date TEXT,
            title TEXT,
            link TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 이벤트 크롤링 및 데이터 저장
def crawl_and_store_events(db_path):
    base_url = 'https://www.jeonju.go.kr'
    event_max_num = get_max_num(db_path, 'events')
    current_page = 1

    while True:
        page_url = f'/planweb/board/list.9is?contentUid=ff8080818990c349018b041a87453954&boardUid=ff8080818b5bc5cf018b6588a3d91e94&page={current_page}&subPath='
        url = base_url + page_url
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', class_='bbs_table')
        rows = table.find_all('tr')[1:]  # 헤더를 제외한 모든 행

        if not rows:
            print("더 이상 데이터가 없습니다. 크롤링 완료.")
            break

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for row in rows:
            num_cell = row.find(class_='check')
            title_cell = row.find(class_='title')
            기간_cell = row.find('td', {'data-cell-header': '기간'})

            if num_cell and title_cell and 기간_cell:
                number = int(num_cell.text.strip())
                title = title_cell.a['title']
                link = base_url + title_cell.a['href']
                기간 = 기간_cell.text.strip().replace('기간', '').strip()

                if number <= event_max_num:
                    print(f"크롤링 중단: 현재 문서 번호 {number}는 이미 존재합니다.")
                    conn.close()
                    return

                content = fetch_content(link, 'view-table')

                # 정규 표현식을 사용하여 필요한 정보 추출
                time = re.search(r"시간\s*(.+?)\s*주최", content)
                age = re.search(r"연령대\s*(.+?)\s*입장료", content)
                ticket = re.search(r"입장료\s*(.+?)\s*장소", content)
                location = re.search(r"장소\s*(.+?)\s*관련사이트", content)

                # 값이 없으면 None으로 처리
                time = time.group(1).strip() if time else None
                age = age.group(1).strip() if age else None
                ticket = ticket.group(1).strip() if ticket else None
                location = location.group(1).strip() if location else None

                # 데이터베이스에 저장
                cursor.execute('''
                    INSERT INTO events (num, title, link, 기간, content, 시간, 연령대, 입장료, 장소) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (number, title, link, 기간, content, time, age, ticket, location))

        conn.commit()
        conn.close()
        current_page += 1
        print(f"{current_page - 1} 페이지 크롤링 완료, 다음 페이지로 이동합니다.")
        
# 뉴스 크롤링 및 데이터 저장
def crawl_and_store_news(db_path):
    base_url = 'https://www.jeonju.go.kr'
    news_max_num = get_max_num(db_path, 'news')
    current_page = 1

    while True:
        page_url = f'/planweb/board/list.9is?page={current_page}&contentUid=ff8080818990c349018b041a87fe3960&boardUid=ff8080818b5bc5cf018ba8ca7216641f&subPath='
        url = base_url + page_url
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', class_='bbs_table')
        rows = table.find_all('tr')[1:]  # 헤더를 제외한 모든 행

        if not rows:
            print("더 이상 데이터가 없습니다. 크롤링 완료.")
            break

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for row in rows:
            num_cell = row.find(class_='num')
            title_cell = row.find(class_='title')
            date_cell = row.find(class_='date')

            if num_cell and title_cell and date_cell:
                number = int(num_cell.text.strip().replace(',', ''))
                title = title_cell.text.strip()
                link = base_url + title_cell.a['href']
                date = date_cell.text.strip()

                if number <= news_max_num or number < 30421:
                    print(f"크롤링 중단: 현재 문서 번호 {number}는 이미 존재하거나 30421보다 작습니다.")
                    conn.close()
                    return

                content = fetch_content(link, 'view-table')

                try:
                    cursor.execute('''
                        INSERT INTO news (num, date, title, link, content) VALUES (?, ?, ?, ?, ?)
                    ''', (number, date, title, link, content))
                except sqlite3.IntegrityError:
                    print(f"중복된 문서 번호 {number}는 저장하지 않습니다.")

        conn.commit()
        conn.close()
        current_page += 1
        print(f"{current_page - 1} 페이지 크롤링 완료, 다음 페이지로 이동합니다.")

def eventjob():
    print("크롤링 작업 시작...")
    crawl_and_store_events(eventdb_path)
    print("크롤링 작업 완료.")
    
def newsjob():
    print("크롤링 작업 시작...")
    crawl_and_store_news(newsdb_path)
    print("크롤링 작업 완료.")
########################################################

# 임베딩 생성
def get_embeddings(texts):
    # 여러 개의 텍스트에 대해 임베딩 생성
    embeddings = [
        genai.embed_content(model='models/embedding-001', content=text, task_type="retrieval_document")["embedding"] 
        for text in texts
    ]
    embeddings_array = np.array(embeddings).astype('float32')
    # print(f"Embeddings shape: {embeddings_array.shape}")  # (n_samples, d) 형식인지 확인

    return embeddings_array

# FAISS 인덱스 생성
def create_faiss_index(embedded_texts):
    d = embedded_texts.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embedded_texts)
    return index

# 걸리는 시간 찾기 -> filtered_travel_times 반환
def get_travel_times(wheretogo):
    conn = sqlite3.connect(timedb_path)  # DB 경로 설정
    cursor = conn.cursor()

    # 결과를 저장할 리스트
    travel_times_list = []

    # 모든 조합을 가져오기 위한 중첩 루프
    for place1 in wheretogo:
        for place2 in wheretogo:
            if place1 != place2:  # 같은 장소는 제외
                # SQL 쿼리 작성
                query = '''
                SELECT pubtrans_duration, car_duration, walk_duration 
                FROM travel_times 
                WHERE start_name = ? AND goal_name = ?
                '''
                cursor.execute(query, (place1, place2))
                result = cursor.fetchone()
                
                if result:
                    pubtrans_duration, car_duration, walk_duration = result
                    travel_times_list.append({
                        'start': place1,
                        'goal': place2,
                        'pubtrans_duration': pubtrans_duration,
                        'car_duration': car_duration,
                        'walk_duration': walk_duration
                    })
                    
                    # 반대 방향 정보 추가
                    travel_times_list.append({
                        'start': place2,
                        'goal': place1,
                        'pubtrans_duration': pubtrans_duration,
                        'car_duration': car_duration,
                        'walk_duration': walk_duration
                    })
                '''else:
                    # DB에 값이 없는 경우 LLM에 요청
                    llm_result = get_travel_time_from_llm(place1, place2)
                    time.sleep(1)
                    travel_times_list.append({
                        'start': place1,
                        'goal': place2,
                        'pubtrans_duration': llm_result['pubtrans_duration'],
                        'car_duration': llm_result['car_duration'],
                        'walk_duration': llm_result['walk_duration']
                    })
                    
                    # LLM 결과를 반대 방향에 추가
                    travel_times_list.append({
                        'start': place2,
                        'goal': place1,
                        'pubtrans_duration': llm_result['pubtrans_duration'],
                        'car_duration': llm_result['car_duration'],
                        'walk_duration': llm_result['walk_duration']
                    })'''
                    
    conn.close()
        
    # '정보 없음'인 항목 제거
    filtered_travel_times = [
        item for item in travel_times_list
        if not (item['pubtrans_duration'] == '정보 없음' and 
                item['car_duration'] == '정보 없음' and 
                item['walk_duration'] == '정보 없음')
    ]

    return filtered_travel_times

def doeventdata():
    create_eventdb(eventdb_path)
    schedule.every().day.at("09:00").do(eventjob) #크롤링하고

def eventduringperiod(travel_start_date, travel_end_date):
    """여행 기간 동안 진행되는 이벤트를 조회."""
    # 데이터베이스 연결
    conn = sqlite3.connect(eventdb_path)  # 적절한 데이터베이스 경로를 설정하세요
    cursor = conn.cursor()
    
    thisevents = []
    cursor.execute("SELECT title, 기간, 시간, 연령대, 입장료, 장소 FROM events")
    events = cursor.fetchall()

    for event in events:
        title, 기간, 시간, 연령대, 입장료, 장소 = event
        
        # 기간 문자열을 분리하고 날짜로 변환
        start_str, end_str = 기간.split('~')
        event_start_date = datetime.strptime(start_str.strip(), '%Y/%m/%d')
        event_end_date = datetime.strptime(end_str.strip(), '%Y/%m/%d')
        event_start_dated = event_start_date.strftime('%Y-%m-%d')
        event_end_dated = event_end_date.strftime('%Y-%m-%d')
        # TODO

        # 여행 기간과 이벤트 기간 비교
        if (event_start_dated <= travel_end_date) and (event_end_dated >= travel_start_date):
            thisevents.append({
                'title': title,
                '기간': 기간,
                '시간': 시간,
                '연령대': 연령대,
                '입장료': 입장료,
                '장소': 장소
            })
    conn.close()
    
    return thisevents

def updatenewsdata():
    create_newsdb(newsdb_path)
    schedule.every().wednesday.at("09:00").do(eventjob)

    conn = sqlite3.connect(newsdb_path)
    cursor = conn.cursor()

    try:
        cursor.execute('ALTER TABLE news ADD COLUMN metadata TEXT')
    except sqlite3.OperationalError:
        pass

    cursor.execute('SELECT id, title, date, content FROM news')
    rows = cursor.fetchall()

    metadata_list = []

    for row in rows:
        id, title, date, content = row
        metadata = {"Type": "news", "title": title, "date": date, "content": content}
        metadata_list.append(metadata)  # 리스트에 추가
        cursor.execute('UPDATE news SET metadata = ? WHERE id = ?', (json.dumps(metadata, ensure_ascii=False), id))

    conn.commit()
    conn.close()

    '''# 메타데이터를 JSON 파일로 저장
    with open('metadata.json', 'w', encoding='utf-8') as json_file:
        json.dump(metadata_list, json_file, ensure_ascii=False, indent=4)  # JSON 파일에 저장

    print("메타데이터가 metadata.json 파일에 저장되었습니다.")'''

    # FAISS 인덱스 초기화
    d = 768  # 임베딩 차원 (모델에 따라 다름)
    index = faiss.IndexFlatIP(d)  # Inner Product 인덱스

    # 메타데이터의 각 제목을 임베딩하고 FAISS에 저장
    titles = [item["title"] for item in metadata_list]
    embeddings = get_embeddings(titles)
    index.add(embeddings)  # FAISS에 벡터 추가
    
    return index, metadata_list, embeddings
    
def query_and_summarize_news(index, metadata_list,embeddings, query):
    query_embedding = get_embeddings([query]) 
    
    # query_embedding의 차원을 확인 및 조정
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)  # (d,) -> (1, d)
    elif query_embedding.ndim > 2:
        raise ValueError("query_embedding의 차원이 너무 높습니다.")

    k = 3  # 가장 가까운 3개 검색
    distances, indices = index.search(query_embedding, k)

    titles_and_summaries = []
    for idx in indices[0]:
        # 코사인 유사도 계산
        cosine_similarity = distances[0][list(indices[0]).index(idx)] / (np.linalg.norm(query_embedding) * np.linalg.norm(embeddings[idx]))
        
        # 코사인 유사도가 0.8 이상인 경우에만 처리
        if cosine_similarity >= 0.8:
            title = metadata_list[idx]['title']
            content = metadata_list[idx]['content']  # 선택된 항목의 content
            
            #LLM에 콘텐츠 전달하여 요약 생성
            summary = summarize_content(content)
            
            titles_and_summaries.append((title, summary))
            #print(f"Title: {title}, Cosine Similarity: {cosine_similarity:.4f}")
            #print(f"Summary: {summary}")  # 요약 출력
        #print("\n")
    return titles_and_summaries

def summarize_content(content):
    prompt_template = f"""
    # 맥락 정보 #
    너는 뉴스 전문을 읽고, 관련 요약을 하는 전문가야.
    특히 공사/행사/축제/개관/폐관 등의 진행기간을 신경써줘
    
    # 지시사항 #
    지금부터 기사의 내용을 읽을거야.
    기사의 내용을 잘 읽고, 진행기간(시간)이나 장소 등을 중심으로 한 줄로 요약해줘.
    만약, 진행기간이나 장소 등에 대한 정보가 없다면 뉴스 내용을 한 줄로 요약해줘.
    
    # 기사 내용 #
    {content}
    
    # 출력 형식 #
    한 문장으로 출력해줘.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{content}")
        ]
    )
    
    chain = prompt | llm

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chain.invoke({"content": content}).content
            return response  # 요약이 성공적으로 생성되면 반환
        
        except Exception as e:
            print(f"오류 발생: {e}")
            if "500" in str(e) or "429" in str(e):  # 내부 서버 오류 또는 쿼트 초과
                wait_time = 2 ** attempt  # 지수 백오프
                print(f"서버 오류, {wait_time}초 후 재시도합니다...")
                time.sleep(wait_time)
    
    return "재시도 한계 초과"  # 최대 재시도 횟수를 초과한 경우


def queryanswer():
    index, metadata_list, embeddings = updatenewsdata()  # 크롤링 및 임베딩
    query1 = '이번달 중에 전주에 대한 특별한 이슈가 있다면 알려주세요?'
    issue=query_and_summarize_news(index, metadata_list, embeddings, query1)  # 쿼리 및 요약
    query2 = '최근 뉴스 중에서 개장, 폐장, 공사 관련 소식이 있나요?'
    aboutplace=query_and_summarize_news(index, metadata_list, embeddings, query2)
    query3 = '여행기간동안 진행하고 있는 행사나 축제에 대한 정보가 있나요?'
    aboutevent=query_and_summarize_news(index, metadata_list, embeddings, query3)
    
    return issue, aboutplace, aboutevent

def generate_itinerary(wantlocations, travel_start_date, travel_end_date, cafenum, playnum, morning_start, evening_end, arrival_location, arrival_time, depart_time, depart_location, transport_mode, travelconcept, guitar, trtimes, placedocument):  
    # queryanswer 함수 호출하여 이슈, 장소, 이벤트 정보 받기
    issue, aboutplace, aboutevent = queryanswer()
    thisevents = eventduringperiod(travel_start_date, travel_end_date)
    
    # 필수 변수 초기화
    title = "전주 여행 일정"
    name = "사용자"
    start = travel_start_date
    
    prompt_template = f"""
    # 맥락 정보 #
    너는 전문적인 여행 계획 도우미야. 
    사용자가 전주 지역에서의 여행 일정을 짜는 데 도움을 주고 있어.

    # 지시사항 #
    사용자의 요구사항에 맞춰 최적의 여행 일정을 생성해줘.
    다음 1단계에서 5단계의 모든 과정을 거치며 여행 일정을 생성해줘
    
    1단계 : 아래의 정보는 1순위로 지켜야해!
        - 여행 기간: {travel_start_date}부터 {travel_end_date}까지.
        - 첫날인 {travel_start_date}는 {arrival_location}에서 {arrival_time}에 시작해서 {evening_end}까지 일정을 짜줘
        - 첫날과 마지막날을 제외한 기간에는 {morning_start}에서 {evening_end}까지의 일정을 짜줘
        - 마지막날인 {travel_end_date}는 {morning_start}에 시작해서 {depart_location}에 {depart_time}까지 도착하는 일정을 짜줘
        - 방문하는 모든 장소는 {wantlocations}에 있는 장소만 방문할꺼야.
        - {wantlocations}의 각 장소를 {placedocument}를 참고해 classify를 기억해줘
        - classify가 food인 곳은 하루에 최소 2번, 최대 3번은 일정에 넣어야해.
        - 24시간 기준 8시~10시 사이에 한 곳, 12시~14시 사이에 한 곳, 18시~21시 사이에 한 곳씩 넣어줘.
        - {morning_start}랑 {arrival_time}이 10시보다 늦거나 {depart_time}이 18시 이전이라면 그때는 하루에 두 번만 넣어도 좋아
        - 첫째날 출발과 마지막날 도착을 제외하고 출발/도착은 classify가 hotel인 장소여야 해.
    
    2단계 : 일정을 짤 때 추가 사항들이야.
        - 여행 분위기: {travelconcept}에 맞춰 계획.
        - 신경 써야 할 요소: {guitar}를 신경 써서 일정을 짜줘.
        - 이동 시간: {trtimes}를 참고해 각 장소 간 이동 시간을 찾아.
        - 하루에 classify가 cafe인 곳은 최대 {cafenum}번, classify가 play인 곳은 최대 {playnum}번 배치해줘
        - classify가 같은 장소를 연속해서 가는 것은 최대한 피해.
        - {trtimes} 이동 시간을 고려해. {transport_mode[0]}, {transport_mode[1]} 중 짧은 시간으로 해줘.
        - 짧은 시간으로 선택해주고, 옆에 그에 해당하는 {transport_mode}를 적어줘.
        - 만약 이동 시간 정보가 없다면 기본 이동 시간은 30분이야.
        - 남는 시간은 자유시간으로 생성해도 좋아.
        - 각 장소의 운영 시간과 날씨를 고려해 조정해줘.

    3단계: 아래의 방법을 통해 출력의 마지막에 여행과 관련된 정보를 제공해줘.
        - {issue}를 참고해 전주 이슈 알려줘.
        - {aboutplace}와 관련된 장소가 있으면 알려줘.
        - {aboutevent}를 참고해 행사/축제 정보를 알려줘.
        - {thisevents}의 제목, 기간, 시간, 연령대, 입장료, 장소를 포함해.

    4단계: 출력 형식은 다음과 같아. 반드시 아래의 예시 형식을 꼭 지켜줘.
        - {wantlocations}의 장소들을 가는 일정표를 작성할건데 1단계와 2단계를 지켜서 일정을 만들어줘.
        - 일정표에는 '|날짜|출발지|출발시간|이동수단|이동시간|도착시간|도착지|도착지category|도착지에머무는시간'의 형식으로 제공해줘
        - 각 장소에는 1~2시간 정도 머무는 것도 고려해서 시간을 계산해줘
        - 도착지 category는 {placedocument}를 참고해줘.
        - 3단계의 관련 전주 이슈, 행사/축제, 추가정보가 있을 경우에만 모든 일정표가 끝나고 출력의 제일 하단에 정보와 이유를 함께 알려줘.

    5단계:여행 일정표를 생성해줘!
    
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{wantlocations}", "{travel_start_date}","{travel_end_date}","{cafenum}","{playnum}","{morning_start}","{evening_end}","{arrival_location}","{arrival_time}","{depart_time}","{depart_location}","{transport_mode}","{travelconcept}","{guitar}","{trtimes}","{placedocument}")
        ]
    )
    chain = prompt | llm
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = chain.invoke({
                "title": title,
                "name": name,
                "start": start,
                "wantlocations": wantlocations,
                "travel_start_date": travel_start_date,
                "travel_end_date": travel_end_date,
                "cafenum": cafenum,
                "playnum": playnum,
                "morning_start": morning_start,
                "evening_end": evening_end,
                "arrival_location": arrival_location,
                "arrival_time": arrival_time,
                "depart_time": depart_time,
                "depart_location": depart_location,
                "transport_mode": transport_mode,
                "travelconcept": travelconcept,
                "guitar": guitar,
                "trtimes": trtimes,
                "placedocument": placedocument,
                "issue": issue,
                "aboutplace": aboutplace,
                "aboutevent": aboutevent,
                "thisevents": thisevents
            }).content
            
            return response
        
        except Exception as e:
            print(f"오류 발생: {e}")
            if "500" in str(e) or "429" in str(e):
                wait_time = 2 ** attempt
                print(f"서버 오류, {wait_time}초 후 재시도합니다...")
                time.sleep(wait_time)
    
    return "재시도 한계 초과"


def get_user_input():
    # 여행 기간 입력
    travel_start_date = input("여행 시작 날짜 (2024-10-09):") 
    travel_end_date = input("여행 끝나는 날짜 (2024-10-09):")
    
    # 하루에 방문할 카페와 놀이시설 수 입력
    cafenum = int(input("하루에 방문할 카페의 최대 수를 입력하세요: "))
    playnum = int(input("하루에 방문할 놀이시설의 최대 수를 입력하세요: "))
    
    # 가고 싶은 장소 입력
    # 최소 하루에 밥2번 + play,cafe 3개 -> 5개
    wantlocations = input("가고 싶은 장소를 입력하세요 (쉼표로 구분): ").split(',')
    wantlocations = [loc.strip() for loc in wantlocations]
    
    # 여행 분위기/난이도 입력
    travelconcept = input("어떤 분위기/난이도의 여행을 원하시나요? (예: 편안한, 모험적인 등): ")
    
    # 여행 첫날 도착 장소 및 시간 입력
    arrival_location = input("여행 첫날 도착할 장소를 입력하세요: ")
    arrival_time = input("여행 첫날 도착 시간을 입력하세요 (24시간 기준): ")
    
    # 여행 마지막날 시간 입력
    depart_location = input("여행 마지막날 출발 장소를 입력하세요: ")
    depart_time = input("여행 마지막날 출발 시간을 입력하세요 (24시간 기준):")
    
    # 여행 활동 시간 입력
    morning_start = input("여행 활동 시작 시간 (오전, 24시간 기준): ")
    evening_end = input("여행 활동 종료 시간 (오후, 24시간 기준): ")
    
    #그 외 고려할 사항
    guitar = input("따로 신경써 줬으면 하는 부분은? : ")
    
    # 이동 수단 선택
    print("이동수단을 선택하세요:")
    print("1: 대중교통 + 걷기")
    print("2: 자동차 + 걷기")
    transport_choice = int(input("1 또는 2를 입력하세요: "))
    
    if transport_choice == 1:
        transport_mode = ( "pubtrans_duration", "walk_duration" )
    elif transport_choice == 2:
        transport_mode = ( "car_duration", "walk_duration" )
    else:
        print("잘못된 선택입니다. 기본값으로 대중교통 + 걷기를 선택합니다.")
        transport_mode = ( "pubtrans_duration", "walk_duration" )
    
    return {
        "wantlocations": wantlocations,
        "travel_start_date": travel_start_date,
        "travel_end_date": travel_end_date,
        "travelconcept": travelconcept,
        "arrival_location": arrival_location,
        "arrival_time": arrival_time,
        "depart_location": depart_location,
        "depart_time": depart_time,
        "cafenum": cafenum,
        "playnum": playnum,
        "morning_start": morning_start,
        "evening_end": evening_end,
        "guitar" : guitar,
        "transport_mode": transport_mode
    }

def plan_trip():
    user_input = get_user_input()
    destinations = user_input["wantlocations"]
    travel_start_date_str = user_input["travel_start_date"]  # 여행 시작일 (문자열)
    travel_end_date_str = user_input["travel_end_date"] 
    cafenum = user_input["cafenum"]
    playnum = user_input["playnum"]
    morning_start = user_input["morning_start"]
    evening_end = user_input["evening_end"]
    arrival_location = user_input["arrival_location"]
    arrival_time = user_input["arrival_time"]
    depart_location = user_input["depart_location"]
    depart_time = user_input["depart_time"]
    transport_mode = user_input["transport_mode"]
    travelconcept = user_input["travelconcept"]
    guitar = user_input["guitar"]
    
    travel_start_date = datetime.strptime(travel_start_date_str, '%Y-%m-%d')
    travel_end_date = datetime.strptime(travel_end_date_str, '%Y-%m-%d')
    travel_start_dated = travel_start_date.strftime('%Y-%m-%d')
    travel_end_dated = travel_end_date.strftime('%Y-%m-%d')
    print("여행일자")
    print(travel_start_dated, travel_end_dated)
    eventduringperiod(travel_start_dated, travel_end_dated)
    
    trtimes = get_travel_times(destinations)
    placedocument = placedoc(placedb_path, destinations)
    #print("placedocument 출력")
    #print(placedocument)


    # LLM을 통해 최적의 일정 생성
    itinerary = generate_itinerary(destinations, travel_start_dated, travel_end_dated, cafenum, playnum, 
                                    morning_start, evening_end, arrival_location, 
                                    arrival_time, depart_location, 
                                    depart_time, transport_mode, travelconcept, guitar, trtimes, placedocument)
    

    # 관련 정보 출력
    print("최적의 여행 일정:")
    print(itinerary)
    
def gettimetable():
    doeventdata
    updatenewsdata()
    queryanswer()
    plan_trip()
    
if __name__ == "__main__":
    
    gettimetable()


'''while True:
    schedule.run_pending()
    time.sleep(1)  # 1초 대기'''
    

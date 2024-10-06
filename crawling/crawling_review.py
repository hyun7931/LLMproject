import pandas as pd
import time
import re

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# CSV 파일 로드
df = pd.read_csv('./cafe_review.csv')
print(df)

# 수집할 정보들 
review_json = {}    # 리뷰 

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")  # GPU 비활성화
# chrome_options.add_argument("--headless")  # 창을 띄우지 않고 실행 (필요시 주석 처리)

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.quit()
#df["reviews"] = None

def check_element_exists(driver, xpath):
    try:
        # 요소를 찾을 수 있는지 시도
        element = driver.find_element(By.XPATH, xpath)
        print("Buttons .")
        return True
    except NoSuchElementException:
        # 요소를 찾을 수 없는 경우
        print("No Buttons.")
        return False
    
index_input = input("저장할 인덱스 번호를 입력하세요: ")
    
index = int(index_input)

for i in range(index, len(df)): 
    
    print('======================================================') 
    print(f'{i+1}번째 식당: ', df['cafe_name'][i]) 

    driver = webdriver.Chrome(service=service, options=chrome_options)    
    # 식당 리뷰 개별 url 접속
    driver.get(df['naver_map_url'][i]) 
    time.sleep(2) 

    review_list = [] 
    count = 0


    # 더보기 버튼 다 누르기 (10개씩 나옴)
    while True: 
        exit_while = 0

        # 스크롤을 끝까지 내리는 부분
        scroll_element = driver.find_element(By.TAG_NAME, "body")
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            # 스크롤을 끝까지 내림
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            # 새로운 높이를 가져옴
            new_height = driver.execute_script("return document.body.scrollHeight")

            # 더 이상 스크롤이 되지 않으면 중단
            if new_height == last_height:
                break
            last_height = new_height
        print("scrolled to the bottom")

        try: 
            # 더보기 버튼 클릭
            btn = WebDriverWait(driver, 20).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '#app-root > div > div > div > div:nth-child(6) > div:nth-child(3) > div.place_section.k1QQ5 > div.NSTUp > div > a'))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(2) 

        except TimeoutException:
            print('\n-(버튼) 더보기 클릭 버튼이 더 없습니다!-\n')
            exit_while = 1

        # 리뷰 수집
        for j in range(10):

            lreview_btn_path = f'//*[@id="app-root"]/div/div/div/div[6]/div[3]/div[3]/div[1]/ul/li[{count*10 + j + 1}]/div[4]/a[2]'
            is_lreview_btn = check_element_exists(driver, lreview_btn_path)

            lreview_btn2_path = f'//*[@id="app-root"]/div/div/div/div[6]/div[3]/div[3]/div[1]/ul/li[{count*10 + j + 1}]/div[5]/a[2]'
            is_lreview_btn2 = check_element_exists(driver, lreview_btn2_path)

            if(is_lreview_btn):
                lreview_btn = driver.find_element(By.XPATH, lreview_btn_path)
                driver.execute_script("arguments[0].scrollIntoView();", lreview_btn)

                # JavaScript를 이용해 클릭 (강제 클릭)
                driver.execute_script("arguments[0].click();", lreview_btn)
            
                lreview_path = f'//*[@id="app-root"]/div/div/div/div[6]/div[3]/div[3]/div[1]/ul/li[{count*10 + j + 1}]/div[4]/a[1]'
                lreview_text = driver.find_element(By.XPATH, lreview_path).text
                if(lreview_text != ''):
                        review_list.append(lreview_text)
                print("="*50)
                print('#',f'{10*count + j + 1}')
                print(lreview_text)  
            
            elif(is_lreview_btn2):
                lreview_btn2 = driver.find_element(By.XPATH, lreview_btn2_path)
                driver.execute_script("arguments[0].scrollIntoView();", lreview_btn2)

                # JavaScript를 이용해 클릭 (강제 클릭)
                driver.execute_script("arguments[0].click();", lreview_btn2)

                lreview2_path = f'//*[@id="app-root"]/div/div/div/div[6]/div[3]/div[3]/div[1]/ul/li[{count*10 + j + 1}]/div[5]/a[1]'
                lreview2_text = driver.find_element(By.XPATH, lreview2_path).text
                if(lreview2_text != ''):
                        review_list.append(lreview2_text)
                print("="*50)
                print('#',f'{10*count + j + 1}')
                print(lreview2_text)  
            else:          
                try:
                    review_path = f'//*[@id="app-root"]/div/div/div/div[6]/div[3]/div[3]/div[1]/ul/li[{count*10 + j + 1}]'
                    review = driver.find_element(By.XPATH, review_path)

                    time.sleep(2)

                    review_text = review.find_element(By.CSS_SELECTOR, 'div.pui__vn15t2 > a').text
                    if(review_text != ''):
                        review_list.append(review_text)
                    print("="*50)
                    print('#',f'{10*count + j + 1}')
                    print(review_text)                  

                except NoSuchElementException:
                    print('\n-(리뷰) 리뷰의 끝에 도달했습니다!-\n')
                    exit_while = 1
                    break
        
        count += 1

        
        if (exit_while == 1 or count >= 3):
            break
            

    df.at[i, "reviews"] = review_list 
    print(df)

    # 리뷰 총 개수 확인
    review_elements = driver.find_elements(By.CLASS_NAME, 'pui__X35jYm.place_apply_pui.EjjAW')
    review_elements_num = len(review_elements)
    print(f'리뷰 총 개수: {review_elements_num - 10}')

    # 결과를 CSV 파일로 저장
    df.to_csv('./cafe_review.csv', index=False, encoding='utf-8-sig')

    driver.quit()


print(df)


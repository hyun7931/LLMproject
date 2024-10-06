# import package
import pandas as pd
import time
import re

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")  # GPU 비활성화
# chrome_options.add_argument("--headless")  # 창을 띄우지 않고 실행 (필요시 주석 처리)

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)


keyword = '전주 관광지'
url = f'https://map.naver.com/p/search/{keyword}'

df = pd.DataFrame(columns=['site_name', 'naver_map_url'])

driver.get(url)
print("URL: ", url)

driver.switch_to.frame('searchIframe') #  검색하고나서 가게정보창이 바로 안뜨는 경우 고려해서 무조건 맨위에 가게 링크 클릭하게 설정
driver.implicitly_wait(3)
#print(driver.page_source)

def check_pages(driver, page_btn_path):
    try:
        # 요소를 찾을 수 있는지 시도
        page_btn = driver.find_element(By.CSS_SELECTOR, page_btn_path)
        element = page_btn.find_elements(By.CLASS_NAME, 'mBN2s')
        count = len(element)
        print()
        print("page # :", count)
        print
        return count
    except NoSuchElementException:
        # 요소를 찾을 수 없는 경우
        print()
        print("No pages !")
        print()
        return 0

page_btn_path = '#app-root > div > div.XUrfU > div.zRM9F'
page_num = check_pages(driver, page_btn_path)

for i in range (page_num):
    scroll_element = driver.find_element(By.XPATH,'//*[@id="_pcmap_list_scroll_container"]')
    # 스크롤이 가능할 때까지 반복
    last_height = driver.execute_script("return arguments[0].scrollHeight", scroll_element)

    while True:
        # 컨테이너 내부 스크롤 600 픽셀 내리기
        driver.execute_script("arguments[0].scrollTop += 3000;", scroll_element)
        
        # 페이지가 로딩될 시간을 주기 위해 잠시 대기
        time.sleep(1) 

        # 새로운 높이를 가져와 비교
        new_height = driver.execute_script("return arguments[0].scrollHeight", scroll_element)
        # 스크롤이 완료되었는지 확인
        if new_height == last_height:
            break

        last_height = new_height
    
    span_elements = scroll_element.find_elements(By.CLASS_NAME, "YwYLL")
    time.sleep(2)
    span_text = [span.text for span in span_elements]
    span_text_len = len(span_text)
    # 리스트의 길이 확인

    print("Beginning new page #", i , " !")
    print("places # in this page : ", span_text_len)
    print("="*50)

    for j in range(span_text_len):
        # div[1]을 찾은 후, a 태그를 ID로 접근
        element_xpath = f'//*[@id="_pcmap_list_scroll_container"]/ul/li[{j + 1}]'
        place_element = driver.find_element(By.XPATH, element_xpath)
        driver.execute_script("arguments[0].scrollIntoView(true);", place_element)

        # 해당 div 안의 a 태그를 ID로 찾음 (만약 ID가 있다면)
        place_element_a = place_element.find_element(By.CLASS_NAME, "P7gyV")  # a 태그 ID에 맞게 변경
        place_element_a.click()
        time.sleep(2)

        cu = driver.current_url # 검색이 성공된 플레이스에 대한 개별 페이지 
        print(cu)
        res_code = re.findall(r"place/(\d+)", cu)
        final_url = 'https://pcmap.place.naver.com/restaurant/'+res_code[0]+'/review/visitor#' 
        
        print(j+1, ": ", span_text[j], ', ', final_url)

        new_data = {
            "site_name": span_text[j],
            "naver_map_url": final_url        
        }

        # DataFrame에 새 행 추가
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

        driver.back()  # 이전 페이지로 돌아가기
        driver.switch_to.default_content()
        driver.switch_to.frame('searchIframe')  # 다시 iframe 전환
        driver.implicitly_wait(3) # 대기
    
    driver.find_element(By.CSS_SELECTOR, page_btn_path +'> a:nth-child(7)').click()
    time.sleep(2)

    print(df)
    df.to_csv('./sitename.csv', index=False, encoding='utf-8-sig')

print()
print("Finished ! No more pages to crawl !")
print()



# 최종 데이터프레임 출력 및 CSV 파일 저장
driver.quit()


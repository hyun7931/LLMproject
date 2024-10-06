# import package
import pandas as pd
import time
import re

from selenium.webdriver.common.action_chains import ActionChains
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")  # GPU 비활성화
# chrome_options.add_argument("--headless")  # 창을 띄우지 않고 실행 (필요시 주석 처리)

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome()


keyword = '전주맛집'
url = f'https://map.naver.com/p/search/{keyword}'

df = pd.DataFrame(columns=['restaurant_name', 'naver_map_url'])

# id="app-root"인 div 요소 찾기
driver.get(url)
print("URL: ", url)

driver.switch_to.frame('searchIframe') #  검색하고나서 가게정보창이 바로 안뜨는 경우 고려해서 무조건 맨위에 가게 링크 클릭하게 설정
driver.implicitly_wait(3)
#print(driver.page_source)

scroll_element = driver.find_element(By.XPATH,'//*[@id="_pcmap_list_scroll_container"]')
# 스크롤이 가능할 때까지 반복
last_height = driver.execute_script("return arguments[0].scrollHeight", scroll_element)

driver.implicitly_wait(3)
span_elements = driver.find_elements(By.CSS_SELECTOR, "span.place_bluelink.TYaxT")

# 각 요소의 텍스트 추출
for i in range(last_height):
    element_xpath = f'//*[@id="_pcmap_list_scroll_container"]/ul/li[{i + 1}]'  # 1-based index
    place_element = driver.find_element(By.XPATH, element_xpath)
    place_element_path = place_element.find_element(By.XPATH, 'div[1]/a')
    place_element_path.click()

    time.sleep(5)

    cu = driver.current_url # 검색이 성공된 플레이스에 대한 개별 페이지 
    print(cu)
    res_code = re.findall(r"place/(\d+)", cu)
    final_url = 'https://pcmap.place.naver.com/restaurant/'+res_code[0]+'/review/visitor#' 
    
    print(i, ": ", final_url)
    print("="*20)
    df.loc[i, "naver_map_url"]=final_url 

    driver.back()  # 이전 페이지로 돌아가기
    driver.switch_to.default_content()
    driver.switch_to.frame('searchIframe')  # 다시 iframe 전환
    driver.implicitly_wait(3) # 대기

# 최종 데이터프레임 출력 및 CSV 파일 저장
print(df)
driver.quit()

df.to_csv('./filename.csv', index=False, encoding='utf-8-sig')

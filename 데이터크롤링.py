import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from category_encoders import OrdinalEncoder

import pickle
import csv
import json
import sqlite3


'''
스크래핑 함수 작성
'''
# 탑텐몰 사이트 크롤링
BASE_URL = f"topten10mall.com/kr/front/"
brand_list = ['ziozia','andz','olzen', 'edition', 'topten','polham','projectm','tmaker']

# 사이트 연결
def get_page(page_url):
    page = requests.get(page_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup, page

# 페이지 수 넣으면, 해당페이지의 옷 목록(clot_list에) 저장
def get_reviews(page_num=1): 
    clot_list=[]
    for r in brand_list:
        review_url = f"https://{r}.{BASE_URL}search/categorySearch.do?ctgNo=1001&currentPage={page_num}&rowsperPage=30&sort=saleCnt&statusCd=&lCateId=&cateId=&partner=&color=&size=&season=&minPrice=&maxPrice=&recommendDispYn=Y#"
        soup = get_page(review_url)[0]
        names = soup.find('div', id='searchGoods').find_all('p', class_='card-goods__text')

        for i in names:
            clot_list.append(i.text)

    return clot_list

# 1페이지 ~ page_num까지 all_list에 목록저장(페이지 10 넘는곳은 없음)
def scrape_by_page_num(page_num=2):
    all_list = []
    for i in range(1, page_num+1):                
        all_list.extend(get_reviews(i))

    return all_list

# 페이지랑 상관없이 원하는 개수만큼만 목록 가져옴
def scrape_by_review_num(clot_num):
    reviews = []  
    page_num = (clot_num//30) +1   # 한 페이지에 10개씩 표시되서 0~9까지는 1페이지, 10~19는 2페이지
    reviews = scrape_by_page_num(page_num)[:clot_num] # review_num이 10이면 page_num이 2가 되서 [:review_num]으로 개수 잡아줌

    return reviews

# 데이터 뽑아오기
# 데이터 받을 때 꼭 튜플로 받아야 dataframe으로 만들기 편함!!! [(1,2)]이런식으로 받아야함.. [{1:2}]처럼 dict로 받으면 바꾸기 힘듬
# 옷 이름에 점수 추가 append 할 때, (a,b)형식의 튜플로 지정해야 dataframe형식으로 변환 가능, {a:b}형식 절대 xx
last_list = []
last_list.extend(scrape_by_page_num(10))


''''
크롤링 데이터 분류작업 (티셔츠, 맨투맨, 후드티, 자켓, 패딩, 셔츠, 니트, 가디건, 코트, 바지)
'''
# 티셔츠
t1 = [s for s in last_list if '티셔츠' in s]
t2 = [s for s in last_list if '반팔티' in s]
t3 = [s for s in last_list if '긴팔티' in s]
t4 = [s for s in last_list if '반팔 티' in s]
t5 = [s for s in last_list if '긴팔 티' in s]
t_shirts = t1+t2+t3+t4+t5

# 맨투맨
mentomen = [s for s in last_list if '맨투맨' in s]

# 후드
hude = [s for s in last_list if '후드' in s]

# 자켓
zacket = [s for s in last_list if '자켓' in s]

# 패딩
paka = [s for s in last_list if '파카' in s]
padding = [s for s in last_list if '패딩' in s]
padding = padding+paka

# 셔츠
s1 = [s for s in last_list if '셔츠' in s]
s2 = [s for s in last_list if '셔츠형' in s]

# a_sub_b = [x for x in a if x not in b] # list a에서 b를 빼기
s1_s2 = [x for x in s1 if x not in s2]
shirts = [x for x in s1_s2 if x not in t1]

# 니트
knit = [s for s in last_list if '니트' in s]

# 가디건
cardigan = [s for s in last_list if '가디건' in s]

# 코트
coat = [s for s in last_list if '코트' in s]

# 바지
slace = [s for s in last_list if '슬랙스' in s]
pants =  [s for s in last_list if '팬츠' in s]
pants = pants+ slace



'''
평점을 스크래핑 하는 부분은 제약이 있어서 분류된 데이터를 카테고리화 하고 평점은 가상데이터로 만듬
'''
tt = []
for i in range(len(t_shirts)):
    tt.append((t_shirts[i], random.randint(1,5)))
tt = pd.DataFrame(tt, columns=['이름', '평점'])
tt['종류'] = '티셔츠'

mt = []
for i in range(len(mentomen)):
    mt.append((mentomen[i], random.randint(1,5)))
mt = pd.DataFrame(mt, columns=['이름', '평점'])
mt['종류'] = '맨투맨'

hd = []
for i in range(len(hude)):
    hd.append((hude[i], random.randint(1,5)))
hd = pd.DataFrame(hd, columns=['이름', '평점'])
hd['종류'] = '후드'

zc = []
for i in range(len(zacket)):
    zc.append((zacket[i], random.randint(1,5)))
zc = pd.DataFrame(zc, columns=['이름', '평점'])
zc['종류'] = '자켓'

pk = []
for i in range(len(padding)):
    pk.append((padding[i], random.randint(1,5)))
pk = pd.DataFrame(pk, columns=['이름', '평점'])
pk['종류'] = '패딩'

st = []
for i in range(len(shirts)):
    st.append((shirts[i], random.randint(1,5)))
st = pd.DataFrame(st, columns=['이름', '평점'])
st['종류'] = '셔츠'

kn = []
for i in range(len(knit)):
    kn.append((knit[i], random.randint(1,5)))
kn = pd.DataFrame(kn, columns=['이름', '평점'])
kn['종류'] = '니트'

ca = []
for i in range(len(cardigan)):
    ca.append((cardigan[i], random.randint(1,5)))
ca = pd.DataFrame(ca, columns=['이름', '평점'])
ca['종류'] = '가디건'

co = []
for i in range(len(coat)):
    co.append((coat[i], random.randint(1,5)))
co = pd.DataFrame(co, columns=['이름', '평점'])
co['종류'] = '코트'

pt = []
for i in range(len(pants)):
    pt.append((pants[i], random.randint(1,5)))
pt = pd.DataFrame(pt, columns=['이름', '평점'])
pt['종류'] = '바지'


'''
데이터 병합 및 DB화
'''
# 전체 데이터 = category
category = pd.concat([tt, mt, hd, zc, pk, st,kn, ca, co, pt])
category.reset_index(drop=True, inplace=True)


# DataFrame -> DB(sqlite)에 저장
conn = sqlite3.connect('category.db')
cur = conn.cursor()

cur.execute('DROP TABLE IF EXISTS LIST')
cur.execute('''CREATE TABLE LIST(
    이름 VARCHAR(50),
    평점 INT,
    종류 VARCHAR(50))
    ''')

for i in range(len(category)):
    cur.execute('INSERT INTO LIST (이름, 평점, 종류) VALUES (?, ?, ?)', (category.loc[i][0], category.loc[i][1], category.loc[i][2]))
    
conn.commit()
conn.close()

'''
모델훈련
'''
# 모델훈련
X = category['종류']
y = category['평점']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pipe = make_pipeline(
    OrdinalEncoder(),
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="error",  # error = 1 - accuracy 지표를 사용해 평가합니다.
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=7,
        learning_rate=0.1,
    ),)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# 컬럼 불일치 에러 발생하여 필요한 데이터 예측이 안됨(ValueError: feature_names mismatch:) 
# df => numpy 변경하면 가능해짐
X_train = X_train.to_numpy()
X_test = X_test.values

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(classification_report(y_test, y_pred))

# 카테고리 3번(X_example)을 골랐을 때 평점 1점을 예측
X_example = [[3]]
print(f"리뷰점수 : {pipe.predict(X_example)}")


##### 프로젝트 번외--------------------------------------------------
'''
DataFrame => CSV or JSON
'''
# DataFrame => CSV
category.to_csv('./category.csv', sep=',') # csv파일 만들기

# DataFrame -> json
json_data = category.to_json() 

# json파일 만들기
with open('category.json', 'w') as js:
    json.dump(json_data, js)


"""
머신러닝모델 -> 피클링
"""
# 카테고리 피클 생성
with open('category.pkl', 'wb') as pic:
    pickle.dump(pipe, pic)

# pickle.dump(pipe, open('category.pkl', 'wb'))

# 카테고리 피클 열어서 model에 지정
with open('category.pkl', 'rb') as pic:
    model = pickle.load(pic)
    
# model = pickle.load(open('category.pkl', 'rb'))

# 피클 모델로 예측
# model.predict([4])




# CS_Project3 의류데이터 추천모델
### **프로젝트개요**
- 때때로 어떤 옷을 사야 할지 고민될 때가 있습니다. 
- 그럴 때 제시 된 댄디, 섹시, 남친룩 같은 카테고리를 선택하면 의류를 추천해주는 모델을 만들고자합니다.

### **파이프라인 소개**
1. BeautifulSoup을 이용하여 Top10mall 사이트의 데이터를 수집
2. SQLITE에 수집한 데이터를 적재
3. DB에 적재된 데이터를 바탕으로 대시보드(Metadata) 작성 및 데이터 분석(Python)
4. Heroku를 이용한 배포

<p align="center">
<img src="https://user-images.githubusercontent.com/110000734/225904860-ba25f06e-b1ab-449e-88d3-c52d41f993f3.JPG" width=600>

### **데이터 소개**
- 데이터는 Top10mall의 판매제품의 이름을 모두 스크래핑 하였습니다.
- 판매제품의 범주는 임의로 분류하였습니다.
- 제품평점은 기술부족으로 임의로 지정하였습니다.(가상 데이터)
- 스크래핑 및 데이터 분석은 데이터크롤링.py를 통해 확인가능합니다.
<p align="center">
<img src="https://user-images.githubusercontent.com/110000734/225905545-2746838d-3349-4fe9-bd25-068b007f37b1.JPG" width=550>



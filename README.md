# CS_Project3 의류 추천모델(파이프라인구축)
**해당 프로젝트는 데이터 스크래핑, 적재, 추출, 배포까지 파이프라인 구축에 초점 둔 프로젝트로 완성도가 다소 떨어짐을 미리 안내드립니다.** 
### **프로젝트개요**
- 때때로 어떤 옷을 사야 할지 고민될 때가 있습니다. 
- 그럴 때 제시 된 댄디, 섹시, 남친룩 같은 카테고리를 선택하면 의류를 추천해주는 모델을 만들고자합니다.

### **파이프라인 소개**
1. BeautifulSoup을 이용하여 Top10mall 사이트의 데이터를 수집
2. SQLITE에 수집한 데이터를 적재
3. DB에 적재된 데이터를 바탕으로 대시보드(Tableau) 작성 및 데이터 분석(Python)
4. Heroku를 이용한 배포

<p align="center">
<img src="https://user-images.githubusercontent.com/110000734/225918925-3ea27d4e-a059-4a7f-80dd-08199ec1c0a8.JPG" width=600>

### **데이터 소개**
- 데이터는 Top10mall의 판매제품의 이름을 모두 스크래핑 하였습니다.
- 판매제품의 범주는 임의로 분류하였습니다.
- 제품평점은 기술부족으로 임의로 지정하였습니다.(가상 데이터)
- 스크래핑 및 데이터 분석은 데이터크롤링.py를 통해 확인가능합니다.
<p align="center">
<img src="https://user-images.githubusercontent.com/110000734/225905545-2746838d-3349-4fe9-bd25-068b007f37b1.JPG" width=550>

### 분석 및 배포
**데이터 분석**
- 데이터가 부족하여 의류의 평점을 예측하는 모델에서 accuracy 0.21의 낮은 정확도를 보임
<p align="center">
<img src="https://user-images.githubusercontent.com/110000734/225909270-210f4140-010f-4371-a205-3f7254556ed6.JPG" width=450>

**대시보드 작성**
- 상의가 하의보다 종류가 다양함
- 캐쥬얼룩이 댄디룩보다 평균 평점이 좋았다. 
<p align="center">
<img src="https://user-images.githubusercontent.com/110000734/225933506-b0363b13-9eb6-44bc-a079-1caffca22786.JPG" width=800>

**배포**
- 배포를 하는 것에는 성공하였으나, 페이지이동 구현과정에서 진전하지 못함
<p align="center">
<img src="https://user-images.githubusercontent.com/110000734/225912065-bffab689-8be2-4f1a-8d10-31560b87835b.JPG" width=300>

### 프로젝트 결과 및 회고
- BeautifulSoup을 이용하여 데이터를 스크래핑하고 DB(SQLITE)에 적재하여 분석 및 대시보드를 작성하고 배포하는 일련의 파이프라인을 구축했다.
- 데이터수집에서부터 완성도가 떨어지니 데이터 분석도 엉망이고 대시보드도 작성할만한 것이 없었다.
- 그래도 프로젝트의 목표인 파이프라인 구축에 성공했다는 점을 긍정적으로 생각한다.
- 딥러닝의 부재로 추천시스템이 아닌 분류모델을 만들게 되었다. 추천시스템에 대한 공부가 필요하다.

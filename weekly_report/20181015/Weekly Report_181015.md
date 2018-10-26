# Weekly Report

2018년 10월 15일 월요일 최태민

## Uncertainty in Deep Learning

#### 1. Gaussian Process

- Gaussian Processes for Regression: A Quick Introduction

  - M. Ebden, August 2008
  - 위 논문 참조
- 이전 코드에서의 문제점

  - Covariance Matrix 연산
  - Maximizing likelihood 에서 수식차이
- 결과는 GPflow(파이썬 패키지)와 같음
  - ![Image](/home/taemin/weekly_reports/20181015/Figure_1.png){: width="100%" height="100%"}
  - ![Image](/home/taemin/weekly_reports/20181015/Figure_2.png){: width="100%" height="100%"}
  - ![Image](/home/taemin/weekly_reports/20181015/Figure_3.png){: width="100%" height="100%"}
- 원인 : 가우시안 프로세스에서의 등분산성 가정으로 인해 노이즈의 정도가 균등하에 작동
  - 출처 : [Bayesian machine learning](http://fastml.com/bayesian-machine-learning/) 참고
  - 사진 : ![Image](/home/taemin/weekly_reports/20181015/Figure_5.png){: width="100%" height="100%"}



- 해결 방법 :  Dropout 을 통한 예측값의 분산 자체를 가우시안 프로세스를 통해 학습
  - 추가적으로, 커널함수 변경
    - ![Image](/home/taemin/weekly_reports/20181015/Figure_7.png){: width="100%" height="100%"}
  - 기존 블로그와(Yarin Gal) 참고한 블로그에서 등분산성을 어떻게 해결했는지 확인 중
  - ![Image](/home/taemin/weekly_reports/20181015/Figure_6.png){: width="50%" height="50%"}







   






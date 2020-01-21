#### config: environment
* localhost for testing with sample data
* production for testing in kakao brain
#### Tensorboard 띄우기
* tensorboard --logdir ./logs 
* tensorboard 띄워서 잘 그려지는지 확인

#### 실행
* python main.py (environment) (running mode)
> ex) python main.py localhost train

> ex) python main.py localhost inference

#### sample data 수정
* sample folder안에 있는 csv들 수정

***

### HELP2019에 올리기 
https://help-khidi.kakaobrain.com/
#### 1. 도커로 이미지 빌드
* docker build -t <image_name>:<version_number> .(dockerfile의 위치)
> ex) docker build -t help-test:0.0.1 .

#### 2. 이미지가 제대로 생성되었는지 테스트
* docker run -it <image_name>:<version_number> bash
> ex) docker run -it help-test:0.0.1 /bin/bash/
>> python main.py localhost train  

#### 3. 도커 이미지 파일 저장
* docker save <image_name>:<version_number> | gzip > ./build/<zipped file>
> docker save help-test:0.0.1 | gzip > ./build/help-test.0.0.1.tar.gz
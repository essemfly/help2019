FROM tensorflow/tensorflow:latest-gpu-py3

COPY . /help2019

WORKDIR /help2019

RUN pip install -r requirements.txt
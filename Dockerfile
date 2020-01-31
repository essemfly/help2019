FROM tensorflow/tensorflow:2.0.0-gpu-py3

COPY . /help2019

WORKDIR /help2019

RUN pip install -r requirements.txt
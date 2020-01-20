FROM pytorch/pytorch:latest

COPY . /help2019

WORKDIR /help2019

RUN pip install -r requirements.txt
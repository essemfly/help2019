FROM pytorch/pytorch:latest

COPY src /

WORKDIR /

RUN pip install -r requirements.txt
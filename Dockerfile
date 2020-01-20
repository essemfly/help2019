FROM pytorch/pytorch:latest

COPY src /

WORKDIR /

RUN pip install -r src/requirements.txt
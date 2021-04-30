FROM python:3.6-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

USER root

RUN apt-get update -y && \ 
    apt-get install -y libgomp1

COPY . /opt/train_models/

RUN pip install -r /opt/train_models/requirements.txt

WORKDIR /opt/train_models/



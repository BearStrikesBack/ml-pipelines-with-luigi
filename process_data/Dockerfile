FROM python:3.6-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

USER root

COPY . /opt/process_data/

RUN pip install -r /opt/process_data/requirements.txt

WORKDIR /opt/process_data/



FROM python:3.6-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

USER root

COPY . /opt/make_report/

RUN pip install -r /opt/make_report/requirements.txt

WORKDIR /opt/make_report/


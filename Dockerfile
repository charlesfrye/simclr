ARG PARENT=tensorflow/tensorflow:1.15.0-gpu-py3

FROM ${PARENT}

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY *.py /app/
COPY *.sh /app/
COPY Makefile /app

ENV PYTHONPATH /app

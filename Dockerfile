ARG PARENT=tensorflow/tensorflow:1.15.0-gpu-py3

FROM ${PARENT}

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r requirements.txt
RUN pip install wandb

COPY *.py /app/
COPY Makefile /app

ENV PYTHONPATH /app

ARG WANDB_BASE_URL
ENV WANDB_BASE_URL ${WANDB_BASE_URL}

FROM quay.io/jupyter/scipy-notebook AS base

COPY ./ /home/jovyan/work

WORKDIR /home/jovyan/work

ENV PYTHONPATH="/home/jovyan/work"

RUN --mount=type=cache,target=/root/.cache/pip \
        pip install -r requirements.txt

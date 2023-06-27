FROM tiangolo/uvicorn-gunicorn:python3.8

# Pip dependency Twisted needs gcc
RUN apt-get update
RUN apt-get install -y poppler-utils

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Might be needed to run after Tensorflow, as there is a problem with the latest Numpy at the time of writing.
RUN pip3 uninstall -y numpy
# 1.18.0
RUN pip3 install numpy==1.18.5

RUN pip3 install pydantic==1.9.2
RUN pip3 install protobuf==3.20

WORKDIR "/app"

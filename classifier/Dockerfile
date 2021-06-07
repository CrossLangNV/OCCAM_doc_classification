FROM tiangolo/uvicorn-gunicorn:python3.8

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Might be needed to run after Tensorflow, as there is a problem with the latest Numpy at the time of writing.
RUN pip3 uninstall -y numpy
RUN pip3 install numpy==1.18.5 # 1.18.0

WORKDIR "/app"

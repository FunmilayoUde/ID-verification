FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y cmake

RUN apt-get update && apt-get install -y g++

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN apt-get update && \
    apt-get install -y cmake g++ libglib2.0-0

RUN apt-get update && apt-get install -y tesseract-ocr

ENV CXX=g++

RUN pip install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
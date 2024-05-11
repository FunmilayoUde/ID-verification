FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    gcc \
    libavcodec-dev \
    libswscale-dev \
    libavformat-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libpng-dev \
    libopencv-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    wget \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libtbb-dev \
    libgphoto2-dev \
    ffmpeg \
    usbutils \
    v4l-utils


RUN apt-get update && apt-get install -y v4l-utils

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
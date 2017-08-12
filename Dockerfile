FROM python:3.6.2-slim

RUN useradd -ms /bin/bash serenata_de_amor
WORKDIR /home/serenata_de_amor/rosie

RUN apt-get update && apt-get install -y \
  build-essential \
  libxml2-dev \
  libxslt1-dev \
  python3-dev \
  unzip \
  zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY config.ini.example ./
COPY requirements.txt ./
COPY setup ./
RUN ./setup

USER serenata_de_amor

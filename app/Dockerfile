FROM python:3.11-slim AS base

WORKDIR /project

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY stream_platform ./stream_platform
COPY main_code ./main_code

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt




WORKDIR /project/app

EXPOSE 8000
CMD ["./start.sh"]

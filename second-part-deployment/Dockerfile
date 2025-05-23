FROM python:3.11-slim

LABEL authors="josee"

WORKDIR /app

COPY first-part-model-training .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python"]

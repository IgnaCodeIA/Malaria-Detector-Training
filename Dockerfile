FROM python:3.12

ENV PYTHONUNBUFFERED=True

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

VOLUME /app/models
VOLUME /app/logs

COPY . /app

EXPOSE 6006

CMD ["dvc", "repro"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY recsys ./recsys
COPY app ./app
COPY scripts ./scripts

# The MovieLens data is not baked into the image: mount ./data (see docker-compose.yml)
# after running `python scripts/download_data.py`.
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

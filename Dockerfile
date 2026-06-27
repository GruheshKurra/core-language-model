FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

COPY requirements.txt requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY zyn ./zyn
COPY serve ./serve

EXPOSE 8000

CMD ["sh", "-c", "uvicorn serve.app:app --host ${HOST} --port ${PORT}"]

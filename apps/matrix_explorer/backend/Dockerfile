# NOTE: This file was partially generated using AI assistance.

FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install uv && \
    uv pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
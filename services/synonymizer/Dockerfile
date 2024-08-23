FROM python:3.11-slim

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install -r requirements.txt -U

COPY . .

# Expose the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8081", "--reload"]
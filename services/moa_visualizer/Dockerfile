FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY assets/ ./assets/

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.address", "0.0.0.0"]

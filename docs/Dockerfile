FROM python:3.11-slim as mkdocs-builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git
COPY Makefile .
COPY requirements.txt .
RUN pip install uv && make install
COPY . .
RUN make build

# docs see https://hub.docker.com/r/pierrezemb/gostatic/
FROM pierrezemb/gostatic
COPY --from=mkdocs-builder /app/site /srv/http

ENTRYPOINT ["./goStatic", "-port", "8080"]

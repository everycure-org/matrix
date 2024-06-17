FROM python:3.11-slim as mkdocs-builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git
COPY . .
RUN pip install uv && make build

# docs see https://hub.docker.com/r/pierrezemb/gostatic/
FROM pierrezemb/gostatic
COPY --from=mkdocs-builder /app/site /srv/http

# Use a lightweight Python image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install the high-level framework
RUN pip install --no-cache-dir fastmcp uvicorn

COPY sse_server.py .

EXPOSE 8000

CMD ["python", "sse_server.py"]
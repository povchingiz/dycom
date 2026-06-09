FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY server/ ./server/

# Expose port for web demo
EXPOSE 8000

# Default: run web demo server
# Override with: docker run -e DEMO_PASSWORD=yourpassword facesim
CMD ["python", "server/main.py"]

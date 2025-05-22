FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your app code
COPY . .

# Download and install ngrok
RUN apt-get update && apt-get install -y wget unzip \
    && wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip \
    && unzip ngrok-stable-linux-amd64.zip \
    && mv ngrok /usr/local/bin \
    && rm ngrok-stable-linux-amd64.zip

# Expose Flask port
EXPOSE 5000

# Start Flask and ngrok
CMD ["sh", "-c", "flask run --host=0.0.0.0 --port=5000 & ngrok http 5000 --log=stdout"]

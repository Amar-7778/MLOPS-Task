# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ ./templates/

# Create run directories
RUN mkdir -p /app/runs /app/uploads

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]

# Use an official Python runtime as a parent image
# 3.10-slim is lightweight and sufficient for our needs
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
# build-essential: for compiling C extensions
# git: for cloning dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements to cache them in docker layer
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port (if web service, though currently CLI)
# EXPOSE 8000

# Set the default command to run the CLI help
ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]

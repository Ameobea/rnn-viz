# Use an official Python runtime as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Graphviz and Python dependencies
RUN apt-get update && \
    apt-get install -y graphviz && \
    pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the app with gunicorn when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]

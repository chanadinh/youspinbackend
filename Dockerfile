# Use an official Python runtime as a parent image
FROM python:3.11.10

# Set the working directory in the container
WORKDIR /app
RUN apt-get update && apt-get -y install libgl1

COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Specify the command to run when the container starts
CMD ["python", "app.py"]
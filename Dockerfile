# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the local code to the container 
COPY . .

# Install any needed packages specified in requirement.txt
RUN pip install --no-cache-dir -r requirement.txt

# Make port 8000 visible to the outside world
EXPOSE 8000


# Define environment variable

ENV NAME LLAMA8b

# Run app.py when the container launches
CMD ["python", "app.py"]



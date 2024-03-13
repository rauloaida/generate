# Use an official Python runtime as a base image
FROM python:3.9.6

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
# Ensure you have uvicorn and fastapi in your requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the model directory into the container
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "7860"]

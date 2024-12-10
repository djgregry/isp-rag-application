FROM python

# Change the working directory
WORKDIR /app

# Copy requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code into working directory
COPY server /app

# Expose the port on which the server will run on
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
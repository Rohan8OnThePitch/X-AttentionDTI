# Use an official, lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the cloud server
WORKDIR /app

# Copy your shopping list and install the packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files (including best_model.pt)
COPY . .

# Hugging Face Spaces strictly listen on port 7860
EXPOSE 7860

# Start the Gunicorn server on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "1", "app:app"]
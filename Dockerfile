# Use a lightweight Python base
FROM python:3.10-slim

# Set up the internal server directory
WORKDIR /app

# Copy the shopping list using strict JSON syntax to handle the space
COPY ["project phase 2/requirements.txt", "."]
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the actual code and the 167MB model
COPY ["project phase 2/", "."]

# Hugging Face Spaces strictly require port 7860
EXPOSE 7860

# Start the heavy-duty web server WITH a 120-second timeout
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "1", "--timeout", "120", "app:app"]
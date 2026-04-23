FROM python:3.10-slim
WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8080

# Run the Streamlit application
CMD ["streamlit", "run", "UI.py", "--server.port=8080", "--server.address=0.0.0.0"]

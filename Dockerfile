FROM python:3.11-slim

WORKDIR /app

# Create tmp directories with write permissions
RUN mkdir -p /tmp/chroma_db && chmod 777 /tmp
RUN mkdir -p /tmp/uploads && chmod 777 /tmp/uploads

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", \
     "--server.enableCORS=false"]
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 GRADIO_ANALYTICS_ENABLED=False MPLBACKEND=Agg
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py scoring.py run_alertas.py ./
COPY modelos/ ./modelos/
COPY data/ ./data/
EXPOSE 7860
CMD ["python", "app.py"]

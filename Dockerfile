FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/app.py   ./app.py
COPY main.py      ./main.py

ENV MLFLOW_TRACKING_URI=http://192.168.0.8:5555
ENV MODEL_NAME=network-anomaly-svm
ENV MODEL_STAGE=Production
ENV PORT=8089

EXPOSE 8080

CMD ["python", "app.py"]

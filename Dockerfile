FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/app.py        ./app.py
COPY src/preprocess.py ./preprocess.py

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5555
ENV MODEL_NAME=network-anomaly-svm
ENV MODEL_STAGE=Production
ENV PORT=8089

EXPOSE 8080

CMD ["python", "app.py"]
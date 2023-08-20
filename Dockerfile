FROM python:3.11-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . /usr/src/app

ENV FLASK_APP=server.py

EXPOSE 8080

CMD ["gunicorn" "--workers" "2" "--timeout" "30" "--log-level" "info" "server:app"]
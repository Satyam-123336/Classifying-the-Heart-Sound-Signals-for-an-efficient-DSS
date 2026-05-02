FROM python:3.11-slim

WORKDIR /code

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		ffmpeg \
		libsndfile1 \
		libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
COPY artifacts ./artifacts

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

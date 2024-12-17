FROM python:3.9

WORKDIR /app

COPY flask_app/ /app/

copy models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install -r requirements.txt 

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "app.py"]
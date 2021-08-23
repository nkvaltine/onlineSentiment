FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#get the nltk downloads used in helpers.py
RUN python -m nltk.downloader punkt stopwords wordnet

COPY . .

#for running locally
EXPOSE 8501
CMD [ "streamlit", "run", "./dashboard/onlineSentiment.py"]

# CMD streamlit run ./dashboard/onlineSentiment.py --server.port ${PORT}
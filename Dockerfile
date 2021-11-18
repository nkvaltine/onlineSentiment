FROM python:3.7

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#get the nltk downloads used in helpers.py
#must specify directory otherwise it won't be found.  there are other choices though
RUN python -m nltk.downloader -d /usr/local/nltk_data punkt stopwords wordnet

COPY . .

#for running locally - don't forget you have to bind the port with -p 8501:8501
# EXPOSE 8501
# CMD [ "streamlit", "run", "./onlineSentiment.py"]

CMD streamlit run ./onlineSentiment.py --server.port ${PORT}
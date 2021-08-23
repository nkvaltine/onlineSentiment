
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#must add this to the Dockerfile
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

#tokenize, stop words, lemmatization
def token_lemma(text):
    #TODO: tokenize urls some way
    tokens = word_tokenize(text)
    
    #@ and # are found a lot in tweets and not likely to be useful in my application
    stwords = stopwords.words('english') + ['@', '#']
    filtered_tokens = [word for word in tokens if word not in stwords]

    wnlt = WordNetLemmatizer()
    lemmatized_words = [wnlt.lemmatize(word) for word in filtered_tokens]
    return lemmatized_words

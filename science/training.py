#this script trains and pickles a model for classifying the sentiment of text strings
#input data for training is from a labeled set of tweets - see 'source.txt' in the data folder
#data for classifying is not the same source, so each script will have its own cleaning section
#%%
import pandas as pd
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#must add this to the Dockerfile
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV



#%%
#import and clean data
tweets = pd.read_csv("../data/training.1600000.processed.noemoticon.csv",
                header=None,
                names=['polarity', 'id', 'date', 'query', 'user', 'text'],
                encoding="latin-1")

#temp, for dev
# tweets = tweets.sample(5000, random_state = 11)
#%%

#preprocess

#polarity uses 0 for negative and 4 for positive for some reason or another 🤔
tweets['polarity'] = tweets['polarity'].replace({0: 0, 4: 1})

#these are specific to this data set and won't be useful for what i'm doing
tweets = tweets.drop(columns=['id', 'date', 'query', 'user'])

#drop long tweets
lens = tweets["text"].str.len()
tweets = tweets[lens <= 140]
tweets.reset_index(drop = True, inplace = True)


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



#bag of words
cvect = CountVectorizer(tokenizer=token_lemma,
                        lowercase=True,
                        min_df = 10)
mat = cvect.fit_transform(tweets['text'])
training_set = pd.DataFrame(mat.toarray())


#add word count as a feature
training_set['word_count'] = tweets['text'].apply(len)
labels = tweets['polarity']

#just clean em up a little
# training_set = training_set.reset_index()
# labels = labels.reset_index()

#set up and validate/compare models

# model = LogisticRegression(max_iter = 10000)
# solvers = ['saga', 'liblinear']
# penalty = ['l1', 'l2']
# c_values = [1000, 100, 10, 1.0, 0.1, 0.01, 0.001]
# grid = dict(solver=solvers,penalty=penalty,C=c_values)

model = RandomForestClassifier()
# n_estimators = [20]
# max_features = ['auto']
# max_depth = [30]
# min_samples_split = [2]
# min_samples_leaf = [1]
    # n_estimators = [20, 50, 100]
    # max_features = ['auto', 'sqrt']
    # max_depth = [30, 60, 90, 120]
    # min_samples_split = [2, 5, 10]
    # min_samples_leaf = [1, 2, 4]
n_estimators = [100, 150, 200]
max_features = ['sqrt']
max_depth = [120, 150]
min_samples_split = [2, 5]
min_samples_leaf = [2]

grid = {
    'n_estimators' : n_estimators,
    'max_features' : max_features,
    'max_depth' : max_depth,
    'min_samples_split' : min_samples_split,
    'min_samples_leaf' : min_samples_leaf
}

# define grid search
crossval = RepeatedStratifiedKFold(n_splits=8, n_repeats=3, random_state=1)
grid_search = GridSearchCV(
    estimator=model, param_grid=grid, cv=crossval, \
    scoring='accuracy', error_score=0,  n_jobs=-1, \
    refit=True
    )
grid_result = grid_search.fit(training_set, labels)

# summarize results - we're not saving this data anywhere for later review, but we could?
print("\n\n\n")
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
fittime = grid_result.cv_results_['mean_score_time']
for mean, stdev, param, ft in zip(means, stds, params, fittime):
    print("%f (%f) with: %r in time: %f" % (mean, stdev, param, ft))
#%%



#look at a couple out-of-the-box sentiment analyzers and compare to my model



#pick best model and pickle the output and transformers
best_model = grid_result.best_estimator_
filename = 'model.pkl'
with open(filename, 'wb') as model_file:
    pickle.dump(best_model, model_file)

filename = 'vectorizer.pkl'
with open(filename, 'wb') as vect_file:
    pickle.dump(cvect, vect_file)


#%%
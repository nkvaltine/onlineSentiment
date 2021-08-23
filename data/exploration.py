#This script shows some of the process I use to explore what a data set looks like.
#It's not all inclusive, and is not really intended to be run outside of an interactive enrivonment

#%%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk import FreqDist
from nltk import tokenize
nltk.download('punkt')

#%%

tweets = pd.read_csv("./training.1600000.processed.noemoticon.csv",
                header=None,
                names=['polarity', 'id', 'date', 'query', 'user', 'text'],
                encoding="latin-1")
# %%

#check balance of label types
tweets.polarity.value_counts()
#perfectly balanced!  nice to have a manufactured data set

#polarity uses 0 for negative and 4 for positive for some reason or another 🤔
#let's fix that
tweets.polarity = tweets.polarity.replace({0: 0, 4: 1})


#let's check out the tweet lengths
lens = tweets["text"].str.len()
lens.describe()

#what are these tweets greater than 140 characters?
temp = tweets[(lens > 140)]["text"]
#hmm, some are real, what about longer than 99.5%
junk = lens.quantile(0.995)  #144...
temp = tweets[(lens > junk)]["text"]
#eh, they aren't all bad

#how many do we lose if I just cut off at 140
sum(lens > 140) / len(lens)
#about 1%?  eh, sounds good
tweets = tweets[lens <= 140]

#look at histogram of tweet lengths
lens = tweets["text"].str.len()
plt.hist(lens, bins=60)
#there's some unusual, regularly spaced spikes.  possibly some kind of rounding.
#the pileup at 140 makes sense, tweets are generally limited to 140 characters


#let's check the most common words, see if that makes sense
#this is a bit awkward, but we need one big corpus for this, not a Series of strings
temp = tokenize.word_tokenize(' '.join(tweets['text']))
temp2 = FreqDist(temp)
temp2.plot(100,cumulative=False)
#looks exponential, v. good!

countlist = sorted(dict(temp2).items(), key = lambda it: it[1], reverse = True)
countlist[:20]
#it's punctuation and prepositions.  seems good.  let's check further down
countlist[60:80]
#regular words, not too bad
countlist[800000:800020]
#weird, uncommon stuff. should likely cut out anything with few mentions from the training
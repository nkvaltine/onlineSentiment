#note to self, for running on my Desktop I have to use pyenv to install python 3.7.0 and then install strealit v 0.62.0
#otherwise I get some junk about "illegal hardware instruction"
import streamlit as st
import pickle

from science import classify
import sys
#idk what it is about the import system, but it never seems to just work
sys.path.append("./science")
from helpers import token_lemma

loaded_model = pickle.load(open("./science/model_LR.pkl", 'rb'))
loaded_vectorizer = pickle.load(open("./science/vectorizer_LR.pkl", 'rb'))


model = classify.sentimenter(loaded_model, loaded_vectorizer)

st.title('An online sentiment classifier')

st.write("Enter text, up to 140 characters, that you'd like to know the sentiment of.")
st.write("Sorry, emojis will be ignored 😭")


sentence = st.text_area('Input your text here:', "It was the best of times, it was the worst of times")
maxlen = 140
if len(sentence) > maxlen:
    st.error(f"Input too long by {len(sentence) - maxlen} characters, please try again.")
else:
    result = model.get_probs(sentence)
       
    st.success(f"There's a {result[0]*100:.2f}% chance this is negative and a {result[1]*100:.2f}% chance it's positive.")

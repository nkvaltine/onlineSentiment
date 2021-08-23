import pandas as pd

class sentimenter(object):
    def __init__(self, estimator=None, vectorizer=None):
        self.name = "sentimentAnalyzer"
        self.estimator = estimator
        self.vectorizer = vectorizer

    #expects just a string, not a list of them
    def get_probs(self, sentence: str):
        sentence = sentence[0:140]
        
        #default behavior is to ignore unknown tokens
        mat = self.vectorizer.transform([sentence])
        input = pd.DataFrame(mat.toarray())

        #add word count as a feature
        input['word_count'] = len(sentence)

        output = self.estimator.predict_proba(input)
        return [round(output[0][0], 2), round(output[0][1], 2)]

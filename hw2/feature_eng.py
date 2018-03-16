import os
import json
from csv import DictReader, DictWriter
import nltk.sentiment.util as nlh
import numpy as np
from numpy import array
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
from time import time
def polarity(text):
    """
    Output polarity scores for a text using Vader approach.

    :param text: a text whose polarity has to be evaluated.
    """
    
    vader_analyzer = SentimentIntensityAnalyzer()
    return (vader_analyzer.polarity_scores(text))


SEED = 5

'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is used
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1
        #print (features)
        return features
class NumberSentencesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = ex.count('.')
            i += 1
        #print (features)
        return features

class AvgWordLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = np.mean([len(word) for word in ex.split()])
            i += 1
        #print (features)
        return features

class PosTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i = 0
        for ex in examples:
            
            n = polarity(ex)
            
            #print ("Here:"+str(n))
            
            features[i,0] = n['pos']
            i += 1
        #print (features)
        return features

class NegTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i = 0
        for ex in examples:

            n = polarity(ex)

            #print ("Here:"+str(n))
            
            features[i,0] = n['neg']
            i += 1
        #print (features)
        return features

class NeutralTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i = 0
        for ex in examples:

            n = polarity(ex)

            #print ("Here:"+str(n))
            
            features[i,0] = n['neu']
            i += 1
        #print (features)
        return features

class ExclamationTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i = 0
        count = 0
        for ex in examples:
            #print (ex)
            count = 0
            if ('!' in ex.split()):
                count +=1

            #print ("Here:"+str(n))

            features[i,0] = count
            i += 1
        #print (features)
        return features

class QuotedTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i = 0
        count = 0
        for ex in examples:
            #print (ex)
            count = 0
            if ('\"' in ex.split()):
                count +=1

            #print ("Here:"+str(n))

            features[i,0] = count
            i += 1
        #print (features)
        return features


# TODO: Add custom feature transformers for the movie review data

class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_length', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer()),
            
            ])),
            ('exclamation', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('exclamation', ExclamationTransformer()),
            ])),
            ('quotes', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('quotes', QuotedTransformer()),
            ])),
            ('numbersentences', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('numbersentences', NumberSentencesTransformer()),
            ])),
            ('avgwordlength', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('avgwordlength', AvgWordLengthTransformer()),
            ])),
            ('positive', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('positive', PosTransformer()),
            ])),
            ('negative', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('negative', NegTransformer()),
            ])),
            ('neutral', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('neutral', NeutralTransformer()),
            ])),            
            ('grams', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('grams', CountVectorizer(stop_words='english', ngram_range=(1,2))),

            ])),
            ('tfidf', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('tfidf', TfidfVectorizer()),
            ])), 
           
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)

    feat = Featurizer()

    labels = []
    for l in y_train:
        if not l in labels:
            labels.append(l)

    print("Label set: %s\n" % str(labels))

    # Here we collect the train features
    # The inner dictionary contains certain pieces of the input data that we
    # would like to be able to select with the ItemSelector
    # The text key refers to the plaintext

    feat_full = feat.train_feature({
        'text': [t for t in dataset_x]
    })

    feat_train = feat.train_feature({
        'text': [t for t in X_train]
    })
    # Here we collect the test features
    feat_test = feat.test_feature({
        'text': [t for t in X_test]
    })

    #print(feat_train)
    #print(set(y_train))

    # Train classifier
    parameters = {'alpha':[0.1,0.001,0.0001,1],'loss':['log'],'penalty':['l2'],'max_iter':[15000],'shuffle':[True]}
    lr = SGDClassifier()
    clf = GridSearchCV(lr, parameters)
    t0 = time()
    clf.fit(feat_full,dataset_y)
    print("done in :"+str((time() - t0)))


    print("Best score:" + str(clf.best_score_))
    print("Best parameters set:")
    best_parameters = clf.best_estimator_.get_params()

    best_alpha=(best_parameters['alpha'])
    print ("Best alpha is:"+str(best_alpha))


    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier

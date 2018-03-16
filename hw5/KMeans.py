from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import json
import numpy as np
import collections
class Vectorizer:
    '''
    vectorize text data
    '''
    def __init__(self, max_df=0.5, max_features=10000, min_df=2, use_idf=True):
        self.vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features, min_df=min_df,
                                          stop_words='english', use_idf=use_idf)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)


class DimensionalityReduction:
    '''
    do dimension Reduction on original data vectors
    '''
    def __init__(self, n_components=200):
        # TODO: use make_pipeline to reduce dimension of data
        #.      use SVD in the pipeline to do dimensionality reduction
        #.      use Normalizer to normalize output
        self.svd = TruncatedSVD(n_components)
        self.normalizer = Normalizer(copy=False)
        self.DR = make_pipeline(self.svd, self.normalizer)

    def fit_transform(self, X):
        # TODO: fit and transform data
        return self.DR.fit_transform(X)

class KM:
    '''
    do clustering on dataset
    '''
    def __init__(self, n_clusters, svd, vectorizer, max_iter=100, n_init=1):
        self.n_clusters = n_clusters
        self.svd = svd
        self.vectorizer = vectorizer
        # CHANGE HERE TO USE KMEANS
        self.km = KMeans(n_clusters=n_clusters,init='k-means++',max_iter=max_iter,n_init=n_init,verbose=False)
        
    def fit(self, X):
        #print ("here:")
        #print (X.shape)
        self.feat = X.shape[1]
        # fit your data to do clustering
        self.km.fit(X)

    def print_doc(self):
        # print top topics
        original_space_centroids = self.svd.inverse_transform(self.km.cluster_centers_)
        #original_space_centroids = original_space_centroids.T
        #print (original_space_centroids)
        
        order_centroids = original_space_centroids.argsort()[:, ::-1]

        terms = self.vectorizer.get_feature_names()
        #terms = list(range(0,self.feat))
        #print (terms)
        
        clustering = collections.defaultdict(list)
 
        for idx, label in enumerate(self.km.labels_):
            clustering[label].append(idx)
        return (clustering)
    
    def print_word(self):
        # print top topics
        original_space_centroids = self.svd.inverse_transform(self.km.cluster_centers_)
        #original_space_centroids = original_space_centroids.T
        #print (original_space_centroids)

        order_centroids = original_space_centroids.argsort()[:, ::-1]

        terms = self.vectorizer.get_feature_names()
        #terms = list(range(0,self.feat))
        #print (terms)

        clustering = collections.defaultdict(list)

        for idx, label in enumerate(self.km.labels_):
            clustering[label].append(terms[idx])
        return (clustering)

        
        
        

if __name__ == '__main__':
    with open('../data/20newsgroup.json', 'r') as f:
        # load dataset from json file
        data_string = f.read()
        dataset = json.loads(data_string)

        # get text data and labels
        data = dataset['text']
        labels = dataset['label']
        unique_labels = np.unique(labels)
        labs = np.ndarray.tolist(unique_labels)
        # You will need to do clustering both for words and documents, be careful about whether to use X or X^T
        # You will also need to summarize clustering results
        # What follows is a rough structure, you will need to call them in the right setup

        # vectorize text data
        #print (len(data))
        #features = len(data)
        vectorizer = Vectorizer()
        X = vectorizer.fit_transform(data)
        X_word = X.T
        #print (X.shape) 
        # do dimensionality reduction
        dr = DimensionalityReduction()
        X = dr.fit_transform(X)
        X_word = dr.fit_transform(X_word)
        #print (X.shape)
        # do clustering
        
        km_doc = KM(n_clusters=len(unique_labels), svd=dr.svd, vectorizer=vectorizer.vectorizer)
        km_word = KM(n_clusters=len(unique_labels), svd=dr.svd, vectorizer=vectorizer.vectorizer)

        km_doc.fit(X)
        km_word.fit(X_word)
        clustering = km_doc.print_doc()

        print (clustering)
        clustering = km_word.print_word()
        print (clustering)
        
        
        

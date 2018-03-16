import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
class RNN:
    '''
    RNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, dict_size=5000, example_length=500, embedding_length=32, epoches=3, batch_size=64):
        '''
        initialize RNN model
        :param train_x: training data
        :param train_y: training label
        :param test_x: test data
        :param test_y: test label
        :param epoches:
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.epoches = epoches
        self.example_len = example_length
        self.dict_size = dict_size
        self.embedding_len = embedding_length

        # TODO:preprocess training data
        self.train_x = sequence.pad_sequences(train_x,maxlen=self.example_len)
        self.test_x = sequence.pad_sequences(test_x,maxlen=self.example_len)
        self.train_y = train_y
        self.test_y = test_y
        print (type(self.train_x))
        texts = list()
        
        wid = imdb.get_word_index()
        wid = {k:(v + 3) for k,v in wid.items()}
        wid["<PAD>"] = 0
        wid["<START>"] = 1
        wid["<UNK>"] = 2
        idw = {value:key for key,value in wid.items()}
        for example in np.concatenate((self.train_x,self.test_x),axis=0):
            a=' '.join(idw[i] for i in example)
            texts.append(a)
        #print (self.train_y)

        tokenizer = Tokenizer(5000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        word_index = tokenizer.word_index
        data = sequence.pad_sequences(sequences, maxlen=self.example_len)
        labels = tf.contrib.keras.utils.to_categorical(np.asarray(np.concatenate((self.train_y,self.test_y),axis=0)))
        print (data.shape)
        print (labels.shape)
        

        #make train and test
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        ntest = 25000
        self.train_x = data[:-ntest]
        self.train_y = labels[:-ntest]
        self.test_x = data[-ntest:]
        self.test_y = labels[-ntest:]

        #embedding vectors
        embeddings_index ={}
        f = open('glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((len(word_index) +1, 100))
        for word,i in word_index.items():
            embedding_vector = embeddings_index.get(word)      
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        # TODO:build model
        self.model = Sequential()
        #self.model.add(Embedding(5000,self.embedding_len))

        self.model.add(Embedding(len(word_index)+1,100,weights=[embedding_matrix]))        

        self.model.add(Conv1D(64,5,padding='valid',activation='relu',strides=1))
        self.model.add(MaxPooling1D(pool_size=4))
        self.model.add(LSTM(self.embedding_len,dropout=0.2,recurrent_dropout=0.2))
        self.model.add(Dense(2,activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

        
    def train(self):
        '''
        fit in data and train model
        :return:
        '''
        
        # TODO: fit in data to train your model
        self.model.fit(self.train_x,self.train_y,batch_size=self.batch_size,epochs=self.epoches,validation_data=(self.test_x,self.test_y))

    def evaluate(self):
        '''
        evaluate trained model
        :return:
        '''
        return self.model.evaluate(self.test_x, self.test_y)


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=5000)
    rnn = RNN(train_x, train_y, test_x, test_y)
    rnn.train()
    acc=rnn.evaluate()
    print (acc)


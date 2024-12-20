# Multilabel


from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from functools import reduce
import nltk
nltk.download('punkt')
import string
from nltk.stem import LancasterStemmer
import ast
import spacy
import nltk
import os
import string
import collections
import re
import numpy as np
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
#lancaster = LancasterStemmer()
from nltk.stem.porter import PorterStemmer
from nltk.stem import RegexpStemmer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import (precision_recall_curve, auc, confusion_matrix,
                             f1_score, fbeta_score, precision_score,
                             recall_score, classification_report)
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split

PUNCTUATIONS = string.punctuation

PUNCTUATIONS = PUNCTUATIONS.replace(r'.', '')
PUNCTUATIONS = PUNCTUATIONS.replace(r',', '')
PUNCTUATIONS += '$'+'£'+'&'

STOPWORDS = stopwords.words('english')  # User potrebbe essere stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in STOPWORDS])



def remove_punctuation(text):
    trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
    return text.translate(trans)

def stemSentenceTokenSpace(text, stemmer):
    #Token are defined by space between words
    token_words=text.split()
    stem_text=""
    for word in token_words:
        # Stemming
        stem_text=stem_text+' '+stemmer.stem(word)
    # Return string (as input type)
    return stem_text


def lemmSentence(text):
  doc = load_model(text)
  # Extract the lemma for each token and join
  return " ".join([token.lemma_ for token in doc])


def preprocessing_cleaning(dataset, remove_website, remove_punctuation_bool, remove_number, remove_stopwords_bool, stemming, stemmer, lemmatization):
   #dataset.rename(columns={'0': "index", '1': "Text", '2':'Label'}, inplace = True)
   
   dataset = dataset[['Text', 'Label']]
   
   dataset['preprocessed'] = dataset['Text'].str.lower()
   
   #print(dataset[dataset['preprocessed'].str.contains(r'https?://\S+|www.\.\S+')])

   if remove_website:  # Se si tolgono, poi rimangono rimoste vuote
      dataset['preprocessed']=dataset['preprocessed'].str.replace(r'https?://\S+|www.\.\S+', 'website')

   if remove_punctuation_bool:
      try: # Ho dovuto forzare a stringa altrimenti dava errore perchè input risultava un float
         dataset['preprocessed']=dataset['preprocessed'].map(lambda x: remove_punctuation(str(x)))
      except:
         ipdb.set_trace()
   
   if remove_number:
      dataset['preprocessed']=dataset['preprocessed'].str.replace(r'\d+', 'number', regex=True)
      

   if stemming:
      if remove_stopwords_bool:
         dataset['preprocessed']=dataset['preprocessed'].map(lambda x: remove_stopwords(str(x)))  

      dataset['stem_token_space']=dataset['preprocessed'].map(lambda x: stemSentenceTokenSpace(str(x),stemmer))
   
   if lemmatization:
      dataset['lemm']=dataset['preprocessed'].map(lambda x: lemmSentence(str(x)))

      if remove_stopwords_bool:
         dataset['preprocessed']=dataset['preprocessed'].map(lambda x: remove_stopwords(str(x)))  

   
      
   return dataset

import re

directory_o = '../../shared/Dataset_llm_marco_gian/datasets_preprocessed'
for file in os.listdir(directory_o):
   filename = os.fsdecode(file)
   
   if ((re.search('train', filename)) and ('legal' in filename) or ('cancer' in filename)) :
    print(filename)
    dataset_train = pd.read_csv('../../shared/Dataset_llm_marco_gian/datasets_preprocessed/'+filename)
    dataset_test = pd.read_csv('../../shared/Dataset_llm_marco_gian/datasets_preprocessed/'+filename.replace('trainAll','test'))
# Generating a random multilabel dataset with 5 feature"s and 4 possible labels
    dataset_train['2'] = dataset_train['2'].apply(lambda x: ast.literal_eval(x))
    dataset_test['2'] = dataset_test['2'].apply(lambda x: ast.literal_eval(x))


    mlb = MultiLabelBinarizer()     

# Splitting the dataset into train and test sets
    X_train = dataset_train['1']
    X_test = dataset_test['1']
    y_train = mlb.fit_transform(dataset_train['2'])
    y_test = mlb.fit_transform(dataset_test['2'])

    #print(list(mlb.classes_))


    tfidf_vect = TfidfVectorizer(min_df=1, max_df=0.9, max_features=1000, ngram_range=(1,1))
    X_train = tfidf_vect.fit_transform(dataset_train['1']).toarray()
    X_test = tfidf_vect.transform(dataset_test['1']).toarray()
# Creating the MultiOutput Classifier with Logistic Regression as the base estimator
    classifier = MultiOutputClassifier(DecisionTreeClassifier())

# Fitting the classifier on the training data
    classifier.fit(X_train, y_train)

# Making predictions on the test set
    predictions = classifier.predict(X_test)

# Evaluate the model, for example, using accuracy score
    try:
        accuracy = classifier.score(X_test, y_test)
    except:
       print('Classi diverse train e test')
    print("Accuracy:", accuracy) 





# Generating a random multilabel dataset with 5 features and 4 possible labels
# Creating a multilabel neural network using TensorFlow
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(500, activation='relu', input_shape=(1000,)),  # Input layer with 5 features
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(len(list(mlb.classes_)), activation='sigmoid')  # Output layer with 4 nodes and sigmoid activation for multilabel classification
    ])

# Compiling the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, shuffle = True)

    predictions = classifier.predict(X_test)
    accuracy = classifier.score(X_test, y_test)
    print("Accuracy NN:", accuracy) 
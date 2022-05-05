import os

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import re
import nltk
print('downloading...')
nltk.download('wordnet')
nltk.download('omw-1.4')
print('...done downloading')
import itertools
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def create_pipeline(clf):
    pipeline = Pipeline([('vect', CountVectorizer(ngram_range = (1, 1))),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', clf)])
    return pipeline

def model_metrics(model_dict, X_train, X_test, y_train, y_test):
    metrics_list = []
    for key, clf in model_dict.items():
        pipeline = create_pipeline(clf)
        pipeline.fit(X_train, y_train)
        pred_y = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, pred_y)
        micro_f1 = f1_score(y_test, pred_y, average = 'micro')
        weighted_f1 = f1_score(y_test, pred_y, average = 'weighted')
        macro_f1 = f1_score(y_test, pred_y, average = 'macro')
        metrics = [key, accuracy, micro_f1, weighted_f1, macro_f1]
        metrics_list.append(metrics)
    metrics_df = pd.DataFrame(metrics_list, columns = ['classifier', 'accuracy', 'micro_f1', 'weighted_f1', 'macro_f1'])
    metrics_df.sort_values(by='macro_f1', axis=0,
                    ascending=True, inplace=True)
    metrics_df.reset_index(inplace = True, drop = True)
    return (metrics_df)

def process_text(text_series):
    df = pd.DataFrame(text_series)
    #remove punctuation except the dollar sign
    df['description'] = df['description'].apply(lambda x: re.sub(r'[^\w\s|\$]', '', x))
    #convert to lower case
    df['description'] = df['description'].str.lower()
    #tokenize into sentences and then to words
    df["sentences"] = df['description'].apply(sent_tokenize) 
    df["words"] = df['description'].apply(word_tokenize) 
    #remove stop words
    df["no_stops"] = df["words"].apply(
                                lambda x: [
                                word for word in x if word not in stopwords.words("English")])
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    df["lemmatized"] = df["no_stops"].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
    #remove rare words
    #convert the lemmatized column into one list of tokens
    token_list = list(itertools.chain.from_iterable(df["lemmatized"]))
    #obtain a frequqency distribution of words
    ls = nltk.FreqDist(token_list)
    #pick out all words with a freuqency <2
    rare_words = dict((k, v) for k, v in ls.items() if v <= 2) 
    #delete rare words
    df["processed"] = df["lemmatized"].apply(lambda x: [word for word in x if word not in rare_words.keys()])
    #convert column back into sentence form
    df["processed"] = df["processed"].apply(lambda x: " ".join(x)) 
    return df["processed"] 
import os

import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import config

import re
import nltk
#uncomment the next two lines if running for the first time
# nltk.download('wordnet') 
# nltk.download('omw-1.4')
import itertools
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class PrepareData():
    def __init__(self, df) -> None:
        self.df = df
    
    def drop_duplicates(self):
        processed_data = self.df.drop_duplicates(keep='first')
        return processed_data
    
    def modify(self):
        self.df.rename(columns = config.MODIFY_CLASSES, inplace = True)
        self.df['category'] = self.df['category'].replace(['Billings', 'Internet Problems', 'Poor Customer Service', 'Data Caps', 'Other', 'other'], 
                            [0, 1, 2, 3, 4, 4])
        self.df['state'] = self.df['state'].replace('District of Columbia', 'District Of Columbia')
        return self.df

    def extract(self):
        self.df[["description", "fcc_comments"]] = self.df["description"].str.split(
                                            "- - - - - - - - - - - - - - - - - - - - -",
                                            n=1,expand=True)
        return self.df


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

def deploy_plot(filepath, modelfile = 'lgr_model.pkl'):
        test_input = pd.read_csv(filepath)
        test_input = pd.DataFrame(test_input)
        test_input_variables = test_input['description']
        with open(modelfile, 'rb') as file:
                model = pickle.load(file)
        predictions = model.predict(test_input_variables)
        test_input['predictions'] = predictions
        test_input['predictions'] = test_input['predictions'].replace(
                                        [0, 1, 2, 3, 4], 
                                        ['Billings', 'Internet Problems', 'Poor Customer Service', 'Data Caps', 'Other'])
        test_input.reset_index(drop=True)
        filepath_name = 'datasets/predictions.csv'
        test_input.to_csv (filepath_name, index = False, header=True)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), dpi = 200)
        sns.countplot(
                x='predictions',
                data=test_input,
                dodge=False,
                hue= 'predictions', ax = ax[0])
        ax[0].set_title('Count of Predicted Categories', fontsize=16, fontweight='bold')
        ax[0].set_ylabel('Counts', fontsize=12, fontweight='bold')
        ax[0].set_xlabel('Predictions', fontsize=12, fontweight='bold')
        ax[0].set_xticks([])
        for p in ax[0].patches:
                ax[0].annotate(f'{p.get_height()}', (p.get_x()+0.3, p.get_height()), ha='center', va='top', color='white', size=10)
        ax[0].legend(loc='best')
        grouped_by_data = test_input.groupby(['date', 'predictions'])['predictions'].count()
        grouped_by_data = pd.DataFrame(grouped_by_data)
        grouped_by_data.rename(columns = {'predictions':'number of complaints'}, inplace=True)
        grouped_by_data.reset_index(inplace=True)
        sns.lineplot(data=grouped_by_data.drop(['predictions'], axis = 1),
                x='date',  y='number of complaints', color = 'blue', linewidth=2.5, ci=None, ax = ax[1])
        dates = grouped_by_data.date.unique()
        ax[1].set_xticks([dates[0], dates[len(dates)//2], dates[-1]])
        ax[1].set_xticklabels([dates[0], dates[len(dates)//2], dates[-1]])
        ax[1].set_ylabel('Counts', fontsize=12, fontweight='bold')
        ax[1].set_xlabel('Days', fontsize=12, fontweight='bold')
        ax[1].set_title('Daily Count of Complaints', fontsize=16, fontweight='bold')
        fig.suptitle('Consumer Complaints Tracking', fontsize=20, fontweight='bold')
        plt.savefig('image_files\Model Predictions Report.png')
        plt.show()
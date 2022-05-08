from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from xgboost import XGBClassifier
import model_training_utils

PROCESS_TEXT = FunctionTransformer(model_training_utils.process_text)

CV = StratifiedKFold(n_splits=3)


LGR_PARAMETERS = {
             'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': [True],
             'clf__multi_class': ['auto', 'ovr', 'multinomial'],
             'clf__penalty': ['l2', 'elasticnet'],
             'clf__max_iter': [100, 150, 200]
             }

XGB_PARAMETERS = {
                'vect__ngram_range': [(1, 1)],
                'tfidf__use_idf': [True],
                'clf__n_estimators': [100, 200],
                'clf__subsample':[0.8, 0.9],
                'clf__gamma': [0.25, 0.5, 1],
                'clf__learning_rate': [0.1, 0.20],
                'clf__max_depth': [5, 8],
                'clf__objective': ['logistic']
                }
   

CLASSIFIERS = {'dummy': DummyClassifier(random_state=1607), 'MNB':MultinomialNB(), 
                'RFC':RandomForestClassifier(random_state=1607), 'GBC':GradientBoostingClassifier(random_state=1607), 
                'BGC':BaggingClassifier(random_state=1607), 'LGR':LogisticRegression(random_state=1607),
                'SVM': SGDClassifier(random_state=1607), 'XTC': ExtraTreesClassifier(random_state=1607),
                'XGB': XGBClassifier(random_state=1607)}


LGR_CLF = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(random_state=1607))
            ])

XGB_CLF = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', XGBClassifier(random_state=1607))
            ])

LGR_CLF_NLTK = Pipeline([
            ('text', PROCESS_TEXT),
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(random_state=1607)),
            ])

XGB_CLF_NLTK = Pipeline([
        ('text', PROCESS_TEXT),
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', XGBClassifier(random_state=1607)),
        ])
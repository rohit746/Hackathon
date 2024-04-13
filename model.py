# general
import numpy as np 
import pandas as pd
import dill as pickle
import joblib,os

# text preprocessing
import re
from string import punctuation
import nltk
nltk.download(['stopwords','punkt'])
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

# models
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# metrics
from sklearn.metrics import classification_report

def preprocess(tweet):
  tweet = tweet.lower()
  random_characters = ['â','¢','‚','¬','Â','¦','’',"It's",'Ã','..','Å']
  tokenizer = TweetTokenizer(preserve_case=True, reduce_len=True)
  tweet = tokenizer.tokenize(tweet)
  stopwords_list = set(random_characters+list(punctuation))
  tweet = [word for word in tweet if word not in stopwords_list]
  tweet = re.sub(r'#([^\s]+)', r'\1', " ".join(tweet))
  tweet = re.sub(r'@([^\s]+)', r'\1', "".join(tweet))  
  return tweet

# pickle preprocessing function
model_save_path = "resources/process.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(preprocess,file)


if __name__ == '__main__':

    # load in data
    print('loading in data')
    train = pd.read_csv('https://raw.githubusercontent.com/monicafar147/classification-predict-streamlit-template/master/climate-change-belief-analysis/train.csv')
    print(train.head())
    train['processed'] = train['message'].apply(preprocess)
    print('data preprocessed')
    #print(train.head())

    # train test split
    X = train['processed']
    y = train['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.05, random_state =10)

    #creating a pipeline with the tfid vectorizer and a linear svc model
    svc = Pipeline([('tfidf',TfidfVectorizer()),('classify',LinearSVC())])

    #fitting the model
    svc.fit(X_train, y_train)

    #apply model on test data
    y_pred_svc = svc.predict(X_test)

    #pickle model
    model_save_path = "resources/linear_SVC.pkl"
    with open(model_save_path,'wb') as file:
        pickle.dump(svc,file)

    #creating a pipeline with a tfidf vectorizer and a logistic regression model
    LR_model = Pipeline([('tfidf',TfidfVectorizer()),('classify',(LogisticRegression(C=1.0,solver='lbfgs',random_state=42,max_iter=200)))])

    #fitting the model
    LR_model.fit(X_train, y_train)

    #Apply model on test data
    y_pred_lr = LR_model.predict(X_test)

    #pickle model
    model_save_path = "resources/LR.pkl"
    with open(model_save_path,'wb') as file:
        pickle.dump(svc,file)

    #{'svm__C': 1, 'svm__gamma': 0.01}
    #creating a pipeline with the tfid vectorizer and a linear svc model
    objs = [("tfidf", TfidfVectorizer()),
        ("svm", SVC(kernel="linear", C=1,gamma=0.01))]
    pipe = Pipeline(objs)
    pipe.fit(X_train, y_train)

    #pickle model
    model_save_path = "resources/SVC.pkl"
    with open(model_save_path,'wb') as file:
        pickle.dump(svc,file)

    #apply model on test data
    y_pred_grid = pipe.predict(X_test)

    print('loading LR')
    model_load_path = "resources/LR.pkl"
    
    with open(model_load_path,'rb') as file:
        unpickled_LR = pickle.load(file)

    tweet = "china is to blame for climate change! #die #flood"
    new = preprocess(tweet)
    print(new)
    tweet_pred = unpickled_LR.predict([new])
    print("predicted",tweet_pred)

    print("testing pickling function")
    model_load_path = "resources/process.pkl"
    with open(model_load_path,'rb') as file:
        cleaner = pickle.load(file)
    print(cleaner("this is text @user"))    
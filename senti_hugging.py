import numpy as np
import pandas as pd
import nltk
import regex as re
import string
import statistics
from scipy.stats import ttest_ind
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from transformers import pipeline
CUDA_LAUNCH_BLOCKING=1
import tensorflow as tf
import torch

#download vader from nltk
#nltk.download("vader_lexicon")
#creating an object of sentiment intensity analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#eng_stopwords = stopwords.words('english')
#sentiment_pipeline = pipeline("text-classification", model="siebert/sentiment-roberta-large-english",truncation=True, device=0)
sentiment_pipeline = pipeline("text-classification", model="siebert/sentiment-roberta-large-english",truncation=True, device=0)
#reading csv file
pd.set_option('display.max_colwidth', None)
religious= pd.read_csv("/processed_news/Religious_topics.csv")
ethnic = pd.read_csv("/processed_news/ethnic_topics.csv")

#creates a control group through quasi randomization
sample_base= pd.concat([religious, ethnic])
sample_size_for_quasi=round(sample_base.shape[0]/20)
print(sample_size_for_quasi)
sample_base=sample_base.sample(n=sample_size_for_quasi, random_state=1)
sample_base['scores']=sample_base['body'].apply(lambda body: sentiment_pipeline(str(body)))
print(sample_base['scores'])
sample_base['compound'] = sample_base['scores'].apply(lambda x: x[0]['score'] if x[0]['label'] == 'POSITIVE' else (-x[0]['score'] if x[0]['label'] == 'NEGATIVE' else 0))
sample_base.to_csv("/processed_news/sample_test2.csv",index=False)
#
print("The metrics of the ")
print(sample_base.loc[:, 'compound'].mean())
print(sample_base.loc[:, 'compound'].median())
# Apply the function to each row of the DataFrame
def sentiment_group(df,topics,type):

    df['scores']=df['body'].apply(lambda body: sentiment_pipeline(str(body)))
    df['compound'] = df['scores'].apply(lambda x: x[0]['score'] if x[0]['label'] == 'POSITIVE' else (-x[0]['score'] if x[0]['label'] == 'NEGATIVE' else 0))
    print("{} overall sentiment".format(type))
    print(df.loc[:, 'compound'].mean())
    print(df.loc[:, 'compound'].median())
    print(df['compound'].mean())

    for organname in df['org_name'].unique():
        for to_pic in range(topics):
            org= df.loc[df['org_name'] == organname]

            org_based= org.loc[df['Topic_{}'.format(to_pic+1)]>0.50]
            print("Topic_{} Organization Name: {}".format(to_pic+1,organname))
            print("Sample size: {}".format(org.shape[0]))
            val1=org_based['compound'].mean()
            print("Mean: {}".format(val1))
            val2=org_based.loc[:, 'compound'].median()
            print("Median: ".format(val2))
            print(ttest_ind(org_based['compound'], sample_base['compound'], equal_var=False))
    df.to_csv("processed_news/{}-sent2.csv".format(type),index=False)
sentiment_group(religious,6, "religious")
sentiment_group(ethnic,6, "ethnic")
print("T-test group comparison")
print(ttest_ind(religious['compound'], ethnic['compound'], equal_var=False))

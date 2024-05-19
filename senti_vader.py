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
#creating an object of sentiment intensity analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
eng_stopwords = stopwords.words('english')
#load the model
sia= SentimentIntensityAnalyzer()
#reading csv file
religious= pd.read_csv("processed_news/Religious_topics.csv")
ethnic = pd.read_csv("processed_news/ethnic_topics.csv")
sample_base= pd.concat([religious, ethnic])
#Getting a random sample
sample_base=sample_base.sample(n=737, random_state=1)
#Applying VADER
sample_base['scores']=sample_base['body'].apply(lambda body: sia.polarity_scores(str(body)))
#Getting scores
sample_base['compound']=sample_base['scores'].apply(lambda score_dict:score_dict['compound'])
print("The metrics of the sample")
print("Mean")
print(sample_base.loc[:, 'compound'].mean())
print("Median")
print(sample_base.loc[:, 'compound'].median())
sample_base.to_csv("processed_news/sample_test-vader.csv",index=False) #saving the sample for replication
def sentiment_group(df,topics, type):
    #Applying the model
    df['scores']=df['body'].apply(lambda body: sia.polarity_scores(str(body)))
    #getting the polarity scores
    df['compound']=df['scores'].apply(lambda score_dict:score_dict['compound'])
    #Metrics for the group type overall
    mean_data=df.loc[:, 'compound'].mean()
    print("Mean of the data:", mean_data)
    median_data= df.loc[:, 'compound'].median()
    print("Median of the data:", median_data)
    non_zero=df[df['compound'] != 0]['compound'].mean()
    #Metrics for the different organizations and topics
    for organname in df['org_name'].unique():
        for to_pic in range(topics):
            org= df.loc[df["org_name"] == organname]
            org_based= org.loc[df["Topic_{}".format(to_pic+1)]>0.50]
            org_based["word_count"] = org_based["body"].apply(lambda text: len(text.split()))
            average_word_count = org_based["word_count"].mean()
            #The below is to check if truncation had any effects in the results
            #org_based = org_based[org_based['word_count'] <= 400]
            #Mean and median
            print("Topic_{} Organization Name: {}".format(to_pic+1,organname))
            val1=org_based['compound'].mean()
            print("Mean: {}".format(val1))
            val2=org_based['compound'].median()
            print("Median: ".format(val2))
            #t-test
            print(ttest_ind(org_based['compound'], sample_base['compound'], equal_var=False))
    df.to_csv("processed_news/{}-sent-vader.csv".format(type),index=False) #saves as csv
sentiment_group(religious,6, "religious")
sentiment_group(ethnic,6, "ethnic")
#the comparison test between religious ethnic
print(ttest_ind(religious['compound'], ethnic['compound'], equal_var=False))

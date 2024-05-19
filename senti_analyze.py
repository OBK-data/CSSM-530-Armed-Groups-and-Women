import numpy as np
import pandas as pd
import nltk
import regex as re
import string
import statistics

import statistics
from scipy.stats import ttest_ind
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#download vader from nltk
#nltk.download("vader_lexicon")
#creating an object of sentiment intensity analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
eng_stopwords = stopwords.words('english')

#reading csv file
religious= pd.read_csv("processed_news/religious-sent2.csv")
ethnic = pd.read_csv("processed_news/ethnic-sent2.csv")
sample_base=pd.read_csv("processed_news/sample_test2.csv")
word_count=[]
def sentiment_measure(df,topics):
    frequency = df["org_name"].value_counts()
    print(frequency)
    mean_data=df.loc[:, 'compound'].mean()
    print("Mean of the data:", mean_data)
    median_data= df.loc[:, 'compound'].median()
    print("Median of the data:", median_data)
    non_zero=df[df['compound'] != 0]['compound'].mean()
    print("Number of values between -0.5 and 0.5:", non_zero)
    for organname in df['org_name'].unique():

        for to_pic in range(topics):
            org= df.loc[df["org_name"] == organname]
            org_based= org.loc[df["Topic_{}".format(to_pic+1)]>0.50]

            org_based["word_count"] = org_based["body"].apply(lambda text: len(text.split()))
            average_word_count = org_based["word_count"].mean()
            #This line was used to test if the trunctation could have changed the results
            #org_based = org_based[org_based['word_count'] <= 400]
            print("Topic_{} Organization Name: {}".format(to_pic+1,organname))
            print(org_based.shape[0])
            #Mean value based on topic
            val1=org_based['compound'].mean()
            print("Mean: {}".format(val1))
            #Median value based on topic
            val2=org_based['compound'].median()
            print("Median: {}".format(val2))
            word_count.append(average_word_count)
            print("The average word count of the text within this topic:",average_word_count)
            print(ttest_ind(org_based['compound'], sample_base['compound'], equal_var=True))

sentiment_measure(religious,6)
sentiment_measure(ethnic,6)
print(statistics.mean(word_count))
print(ttest_ind(religious['compound'], ethnic['compound'], equal_var=False))

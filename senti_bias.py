import numpy as np
import pandas as pd
import nltk
import regex as re
import string
import statistics
import pprint
from scipy.stats import ttest_ind
pp = pprint.PrettyPrinter(width=1000)

#combining the files with the news sources with the data that includes sentiment analysis and topic modelling
#For Religious
religious= pd.read_csv("processed_news/religious-sent-vader.csv")#change depending on the sentiment analysis method that you want to use Default is VADER
hamas=pd.read_csv("processed_news/Hamas_whole_updated.csv")
hezbollah=pd.read_csv("processed_news/Hezbollah_whole_updated.csv")
religious2 = pd.concat([hamas, hezbollah], ignore_index=True)
religious2["body"]=religious2["body"].drop_duplicates(keep="first")
religious2=religious2.dropna(subset=['body'])
#replace line breaks if there are any left
religious2["body"]=religious2["body"].replace('\n',' ', regex=True)
religious2["body"] = religious2["body"].str.strip() #lowercase
religious2['body'] = religious2['body'].str.replace(r'\s+', ' ', regex=True) #replace any remaining whitescape
#drop if any duplicates left
religious2.drop_duplicates(subset=["body"])
print(len(religious["body"]))
print(len(religious2["body"]))
#Map the news into the processed data
body_to_news_rel = dict(zip(religious2['body'], religious2['news_source']))
religious['news_source'] = religious['body'].map(body_to_news_rel)

news_counts = religious['news_source'].value_counts()
#Print news sources iteratively
chunk_size = 10
for i in range(0, len(news_counts), chunk_size):
    print(news_counts[i:i + chunk_size])
#For Ethnic
ethnic= pd.read_csv("processed_news/ethnic-sent-vader.csv") #change depending on the sentiment analysis method that you want to use Default is VADER
ltte=pd.read_csv("processed_news/LTTE_whole_updated.csv")
ira=pd.read_csv("processed_news/IRA_whole_updated.csv")
ethnic2 = pd.concat([ltte, ira], ignore_index=True)
ethnic2["body"]=ethnic2["body"].drop_duplicates(keep="first")
ethnic2=ethnic2.dropna(subset=['body'])
#replace line breaks if there are any left
ethnic2["body"]=ethnic2["body"].replace('\n',' ', regex=True)
ethnic2["body"] = ethnic2["body"].str.strip() #lowercase
ethnic2['body'] = ethnic2['body'].str.replace(r'\s+', ' ', regex=True) #replace any remaining whitescape
#drop if any duplicates left
religious2.drop_duplicates(subset=["body"])
sample_base=pd.read_csv("processed_news/sample_test-vader.csv") #change depending on the sentiment analysis method that you want to use Default is VADER
# Map the news_source values from dataframe2 to dataframe1 based on the body column
body_to_news = dict(zip(ethnic2['body'], ethnic2['news_source']))
ethnic['news_source'] = ethnic['body'].map(body_to_news)
news_counts = ethnic['news_source'].value_counts()
for i in range(0, len(news_counts), chunk_size):
    print(news_counts[i:i + chunk_size])
non_western_news=["The Jerusalem Post", "U.P.I.","Wijeya Newspaper Ltd.","HT Media Ltd.","The Associated Newspapers of Ceylon Ltd."]
#Do the analysis without the non-western sources
def sentiment_measure(df,topics):
    #how many organizations are in the sample
    #Checking the data metrics
    df=df[~df['news_source'].isin(non_western_news)] #Remove non-Western sources
    mean_data=df.loc[:, 'compound'].mean()
    print("Mean of the data:", mean_data)
    median_data= df.loc[:, 'compound'].median()
    print("Median of the data:", median_data)
    non_zero=df[df['compound'] != 0]['compound'].mean()
    print("Number of values between -0.5 and 0.5:", non_zero)
    #looping based on organization
    for organname in df['org_name'].unique():
    #looping based on different topics
        for to_pic in range(topics):
            org= df.loc[df["org_name"] == organname]
            org_based= org.loc[df["Topic_{}".format(to_pic+1)]>0.50]

            #org_based["word_count"] = org_based["body"].apply(lambda text: len(text.split()))
            #average_word_count = org_based["word_count"].mean()
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
            #Average word count
            #word_count.append(average_word_count)
            #print("The average word count of the text within this topic:",average_word_count)
            #applying t-test
            print(ttest_ind(org_based['compound'], sample_base['compound'], equal_var=True))

sentiment_measure(religious,6)
sentiment_measure(ethnic,6)
print(ttest_ind(religious['compound'], ethnic['compound'], equal_var=False))

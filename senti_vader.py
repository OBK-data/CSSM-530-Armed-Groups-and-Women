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
#download vader from nltk
#nltk.download("vader_lexicon")
#creating an object of sentiment intensity analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
eng_stopwords = stopwords.words('english')
sia= SentimentIntensityAnalyzer()
def clean_string(text, stem="None"):

    final_string = ""

    # Make lower
    text = text.lower()

    # Remove line breaks
    # Note: that this line can be augmented and used over
    # to replace any characters with nothing or a space
    text = re.sub(r'\n', '', text)

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove stop words
    text = text.split()
    useless_words = stopwords.words("english")
    useless_words = useless_words + ['hi', 'im']

    text_filtered = [word for word in text if not word in useless_words]

    # Remove numbers
    text_filtered = [re.sub(r'\w*\d\w*', '', w) for w in text_filtered]

    # Stem or Lemmatize
    if stem == 'Stem':
        stemmer = PorterStemmer()
        text_stemmed = [stemmer.stem(y) for y in text_filtered]
    elif stem == 'Lem':
        lem = WordNetLemmatizer()
        text_stemmed = [lem.lemmatize(y) for y in text_filtered]
    else:
        text_stemmed = text_filtered

    final_string = ' '.join(text_stemmed)

    return final_string
#reading csv file
religious= pd.read_csv("processed_news\Religious_topics.csv")
ethnic = pd.read_csv("processed_news\ethnic_topics.csv")
sample_base= pd.concat([religious, ethnic])
sample_base=sample_base.sample(n=737, random_state=1)
sample_base['scores']=sample_base['body'].apply(lambda body: sia.polarity_scores(str(body)))
sample_base['compound']=sample_base['scores'].apply(lambda score_dict:score_dict['compound'])
print(sample_base.loc[:, 'compound'].mean())
print(sample_base.loc[:, 'compound'].median())
def sentiment_group(df,topics):
    df['scores']=df['body'].apply(lambda body: sia.polarity_scores(str(body)))

    df['compound']=df['scores'].apply(lambda score_dict:score_dict['compound'])
    mean_data=df.loc[:, 'compound'].mean()
    print("Mean of the data:", mean_data)
    median_data= df.loc[:, 'compound'].median()
    print("Median of the data:", median_data)
    non_zero=df[df['compound'] != 0]['compound'].mean()

    for organname in df['org_name'].unique():

        for to_pic in range(topics):
            org= df.loc[df["org_name"] == organname]
            org_based= org.loc[df["Topic_{}".format(to_pic+1)]>0.50]
            org_based["word_count"] = org_based["body"].apply(lambda text: len(text.split()))
            average_word_count = org_based["word_count"].mean()
            org_based = org_based[org_based['word_count'] <= 400]
            print("Topic_{} Organization Name: {}".format(to_pic+1,organname))
            val1=org_based['compound'].mean()
            print("Mean: {}".format(val1))
            val2=org_based['compound'].median()
            print("Median: ".format(val2))
            print(ttest_ind(org_based['compound'], sample_base['compound'], equal_var=False))
sentiment_group(religious,6)
sentiment_group(ethnic,6)
print(ttest_ind(religious['compound'], ethnic['compound'], equal_var=False))

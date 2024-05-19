# Importing modules
import pandas as pd
import numpy as np
from gensim import utils
import gensim
import gensim.models
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from gensim.test.utils import datapath
from gensim.test.utils import common_corpus, common_dictionary
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pprint import pprint

LTTE= pd.read_csv("processed_news\LTTE_whole.csv")
LTTE["org_name"]="LTTE"
IRA = pd.read_csv("processed_news\IRA_whole.csv")
IRA["org_name"]="IRA"
Hezbollah= pd.read_csv("processed_news\Hezbollah_whole.csv")
Hezbollah["org_name"]="Hezbollah"
Hamas= pd.read_csv("processed_news\Hamas_whole.csv")
Hamas["org_name"]="Hamas"
ethnic= pd.concat([LTTE,IRA])
religious= pd.concat([Hezbollah, Hamas])
stop_words=stopwords.words('english')
removal_words=["would","two","ulster","year","de","la","sri", "belfast", "said", "lanka", "tamil","tiger","eelam","woman","female", "sinn fein","irish republican army","ap","ltte","tamil tigers","gaza","said","jerusalem", "west bank",
"arab","palestinian", "israel","hamas","us","army","police","lankan","people","mr","hezbollah","ira","ireland","irish","catholic","sinn","fein","protestant","republican","northern","israeli","british", " northern ireland",
"united kingdom", "lebanon","israel","palestine","wa","ha","also","west","bank"]
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 700)
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
stop_words.extend(removal_words)
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.max_colwidth = 5000

def lda_diagnose(dataframe,type, number):
    dataframe=dataframe.dropna(subset=['body'])
    dataframe["body"]=dataframe["body"].replace('\n',' ', regex=True)
    dataframe["body"] = dataframe["body"].str.strip()
    dataframe['body'] = dataframe['body'].str.replace(r'\s+', ' ', regex=True)
    dataframe.drop_duplicates(subset=["body"])
    dataframe["pro_body"]=dataframe["body"]
    dataframe["pro_body"]=dataframe["body"].str.lower()
    dataframe["pro_body"] = dataframe["pro_body"].apply(lemmatize_text)
    #dataframe['body'] = dataframe['body'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_words)]))
    datalist = dataframe["pro_body"].values.tolist()
    data_words = list(sent_to_words(datalist))
    data_words = remove_stopwords(data_words)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)# Create Corpus
    texts = data_words# Term Document Frequency
    topicmodel_load= "LDAmodels/{} {}.model".format(type, number)
    lda_model=LdaModel.load(topicmodel_load)
    corpus = [id2word.doc2bow(text) for text in texts]# View
    topics = [lda_model.get_document_topics(text) for text in corpus]
    pprint(lda_model.print_topics())
# Calculate percentage of each topic for each document
    topics_loop = []
    for doc_topics in topics:
        topic_percentages = [0] * number
        for topic, percentage in doc_topics:
            topic_percentages[topic] = percentage
        topics_loop.append(topic_percentages)
    topics_df = pd.DataFrame(topics_loop, columns=[f"Topic_{i+1}" for i in range(number)])
# Create DataFrame from the collected data
    dataframe.reset_index(drop=True, inplace=True)
    topics_df.reset_index(drop=True, inplace=True)
    dataframe_new = pd.concat([dataframe, topics_df], axis=1)
    dataframe_new=dataframe_new.dropna(subset=['body'])
    #dataframe_new.to_csv("processed_news/{}_topics.csv".format(type))

    for organname in dataframe_new['org_name'].unique():
        for to_pic in range(number):
            org_based= dataframe_new.loc[dataframe_new['org_name'] == organname]
            org_based= org_based.sort_values(by=["Topic_{}".format(to_pic+1)], ascending=False)
            print("{} results sorted by Topic {}, Organization Name : {}".format(type,to_pic+1,organname))
            print(org_based["body"].head(10))
            print(org_based["Topic_{}".format(to_pic+1)].head(10))

#Apply the commands to the dataset
lda_diagnose(religious, "Religious",6)
lda_diagnose(ethnic, "Ethnic",6)

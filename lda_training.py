# Importing modules
import pandas as pd
import numpy as np
from gensim import utils
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import LdaModel
import gensim.models
from gensim.test.utils import datapath
from gensim.test.utils import common_corpus, common_dictionary
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pprint import pprint
from nltk.stem import WordNetLemmatizer



#Load the data
LTTE= pd.read_csv("processed_news\LTTE_whole.csv")
IRA = pd.read_csv("processed_news\IRA_whole.csv")
Hezbollah= pd.read_csv("processed_news\Hezbollah_whole.csv")
Hamas= pd.read_csv("processed_news\Hamas_whole.csv")
#Create two different dataframes based on the ethic groups
ethnic= pd.concat([LTTE,IRA])
religious= pd.concat([Hezbollah, Hamas])
#Remove unnecessary word_dist
stop_words=stopwords.words('english')
removal_words=["would","two","ulster","year","de","la","sri", "belfast", "said", "lanka", "tamil","tiger","eelam","woman","female", "sinn fein","irish republican army","ap","ltte","tamil tigers","gaza","said","jerusalem", "west bank",
"arab","palestinian", "israel","hamas","us","army","police","lankan","people","mr","hezbollah","ira","ireland","irish","catholic","sinn","fein","protestant","republican","northern","israeli","british", " northern ireland",
"united kingdom", "lebanon","israel","palestine","wa","ha","also","west","bank"]

stop_words.extend(removal_words)
#Define functions for data cleaning and corpus creation
lemmatizer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
#Create a function for LDA Analysis
def ldamulti(dataframe,type):
    dataframe.body=dataframe.body.str.lower() #lowercase the data
    dataframe['body'] = dataframe.body.apply(lemmatize_text) #lemmatize the text
    #dataframe['body'] = dataframe['body'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_words)]))
    datalist = dataframe.body.values.tolist() #turn words into a list
    data_words = list(sent_to_words(datalist)) #turn sentences into words
    data_words = remove_stopwords(data_words) #remove stopwords
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)# create a corpus
    texts = data_words
    corpus = [id2word.doc2bow(text) for text in texts] #create id2word
    #train models containing 5 to 10 topics
    for i in range(6):
        num_topics=i+5
        if __name__ == '__main__':
            lda_model = LdaModel(corpus=corpus, iterations=50, id2word=id2word, num_topics=num_topics, per_word_topics=True)
            topicmodel_save= "LDAmodels/{} {}.model".format(type,i+5)
            lda_model.save(topicmodel_save)#save model
        #Print the metrics and top 10 keywords, for both of the measures the values that are closer to 0 are better
        print("{} Model {} Topics".format(type,i+6))
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]
        print('\nPerplexity: ', lda_model.log_perplexity(corpus))

ldamulti(ethnic,"Ethnic")
ldamulti(religious,"Religious")

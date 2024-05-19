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
import matplotlib.pyplot as plt
LTTE= pd.read_csv("processed_news\LTTE_whole.csv")
IRA = pd.read_csv("processed_news\IRA_whole.csv")
Hezbollah= pd.read_csv("processed_news\Hezbollah_whole.csv")
Hamas= pd.read_csv("processed_news\Hamas_whole.csv")
ethnic= pd.concat([LTTE,IRA])
religious= pd.concat([Hezbollah, Hamas])
stop_words=stopwords.words('english')
removal_words=["would","two","ulster","year","de","la","sri", "belfast", "said", "lanka", "tamil","tiger","eelam","woman","female", "sinn fein","irish republican army","ap","ltte","tamil tigers","gaza","said","jerusalem", "west bank",
"arab","palestinian", "israel","hamas","us","army","police","lankan","people","mr","hezbollah","ira","ireland","irish","catholic","sinn","fein","protestant","republican","northern","israeli","british", " northern ireland",
"united kingdom", "lebanon","israel","palestine","wa","ha","also","west","bank"]
road="LDAmodels/Ethnic 10.model"
stop_words.extend(removal_words)
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
lemmatizer = WordNetLemmatizer()
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lda_diagnose(dataframe,type):
    dataframe.body=dataframe.body.str.lower()
    dataframe['body'] = dataframe.body.apply(lemmatize_text)
    #dataframe['body'] = dataframe['body'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in (stop_words)]))
    datalist = dataframe.body.values.tolist()
    data_words = list(sent_to_words(datalist))
    data_words = remove_stopwords(data_words)
    # Create Dictionary
    id2word = corpora.Dictionary(data_words)# Create Corpus
    texts = data_words# Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]# View
    coherence=[]
    perplexity=[]
    topic_num=[]
    for i in range(6):
        num_topics=i+5
        topicmodel_load= "LDAmodels/{} {}.model".format(type,i+5)
        lda_model=LdaModel.load(topicmodel_load)
 #Print the Keyword in the 10 topics
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        pprint(lda_model.print_topics())
        doc_lda = lda_model[corpus]
        perplexity_lda=lda_model.log_perplexity(corpus)
        print('\nPerplexity: ', perplexity_lda)  # a measure of how good the model is. lower the better.
        coherence.append(coherence_lda)
        perplexity.append(perplexity_lda)
        topic_num.append("{} topics".format(i+5))

# function to show the plot
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Topic Number')
    ax1.set_ylabel('Perplexity', color=color)
    ax1.plot(topic_num, perplexity, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Coherence', color=color)  # we already handled the x-label with ax1
    ax2.plot(topic_num, coherence, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Evaluation Graph {}'.format(type))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig("Visuals\LDA Graph {}".format(type))
    plt.show()
#def lda_apply(dataframe, type):

 #Compute Coherence Score
lda_diagnose(ethnic,"Ethnic")
lda_diagnose(religious,"Religious")

import stanza
import nltk
import pandas as pd
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
# Download the language model
#stanza.download('en')
#Download dependencies
#nltk.download('punkt')
#build a pipeline
nlp = stanza.Pipeline('en', processors = "tokenize,mwt,pos,lemma,depparse", use_gpu=True, tokenize_no_ssplit=True)
org_name=["Hamas","Hezbollah", "IRA", "LTTE"]
#Create a function
def parsemulti(to_parse):
    lst=[]
    #load Porter Stemmer
    stemmer = PorterStemmer()
    searched_words= ("women","female","woman","she","her","herself","hers")
    searched_words = [stemmer.stem(w.lower()) for w in searched_words] #stem the searched words
    to_parse["body"]=to_parse["body"].drop_duplicates(keep="first") #remove duplicates
    to_parse["sentences"]=to_parse['body'].apply(lambda text: [sent for sent in sent_tokenize(text) if any(True for w in word_tokenize(sent) if stemmer.stem(w.lower()) in searched_words)])#search the stemmed words and filter the sentences that specifically have these keywords
    to_parse=to_parse.explode("sentences") #separete the data to sentences
    to_parse["sentences"]=to_parse["sentences"].str.strip() #strip the sentences
    to_parse["sentences"]=to_parse["sentences"].replace('\n','', regex=True) #remove line breaks if there are any left)
    parse_done=pd.DataFrame(columns=["id", "text","lemma","upos","xpos","feats","head","deprel","start_char","end_char","misc","women","female","woman","she","girl","sister","mother","her","herself","hers"])

    new_df=pd.DataFrame()
    df_list=[]
    #apply the pipeline
    for i in range(to_parse.shape[0]):
        sentence=str(to_parse["sentences"].iloc[i])
        doc = nlp(sentence)
        main_dic = doc.sentences[0].to_dict()#create a dictionary from the results
        parse_dic=main_dic
        #split the dictionary into word relations
        for word in main_dic:
            try:
              #turns words into intelligeble words
              word['head']= str(main_dic[word['head']-1]['text'] if word['head'] > 0 else 'ROOT')
            #passes in cases the word has multiple ids in a sentence,thus no heads
            except KeyError:
              pass
        #save the relations into a dataframe
        new_df= pd.DataFrame.from_dict(parse_dic)
        df_list.append(new_df)
        #prints how much of the text analysis is done
        if i%10==0:
          print(i)
        else:
          continue
    parse_done=pd.concat(df_list)
    print(parse_done)
    #saves the data as csv
    parse_done.to_csv("processed_news/{}_parsed2.csv".format(org_name[num]))

#feeds the data into the pipeline based on organization name
for num in range(len(org_name)):
    newslet= pd.read_csv("processed_news/{}_whole.csv".format(org_name[num]))
    parsemulti(newslet)

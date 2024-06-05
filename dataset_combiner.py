# importing required modules
import pandas as pd
from pypdf import PdfReader
import regex as re
from functools import reduce
import os, os.path
#Lists for organization names and their natures
org_name=["Hamas","Hezbollah","IRA","LTTE"]
org_type=["Religious","Religious","Ethnic","Ethnic"]
rawnews_dir=['raw_news2/' + sub  for sub in org_name]
numb=0
final_df=pd.DataFrame()
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

#goes through the unprocessed news based on the organization
for e in range(len(rawnews_dir)):
    df_list=[]
    path=rawnews_dir[e]
    roll=len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
    print(roll)
    #Concatenates the newspaper pages
    for a in range(roll):
        output=" "
        #turn=a+40 IRA
        turn=a+1
        reader = PdfReader('{}/{} ({}).pdf'.format(rawnews_dir[e],org_name[e],turn))
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text=page.extract_text()
            text = text.replace('(Page \d+ of \d+\n.*+)', '\n')
            output+=text
#Removes redundant text such as copyright, page number etc.
        output = output.replace('| About LexisNexis | Privacy Policy | Terms & Conditions | Copyright © 2024 LexisNexis\n', '')
        output = re.sub((r"Page \d+ of \d+\n.*"), '', output)
        #Takes the body of the text which is in between the "Body" header and "End of the Document"
        body=re.findall(r"Body\n([\s\S]+?)(?=End of Document)|Graphic\n([\s\S]+?)(?=End of Document)", output)
        body = [list(filter(None, lst)) for lst in body]
        body = [reduce(lambda x, y: x + y, lst) for lst in body]
#Extracting headlines while accounting for potential mistakes

        # Extract headlines
        headlines = re.findall(r"\d+\.\s*(.*?)(?=Client\/)", output, re.DOTALL)
        headlines = [s.rstrip() for s in headlines]

        # Build patterns for each headline to capture the news source
        head_patterns = [sub + r"\n(.*)\n(?=\nClient/Matter: -None-|$)" for sub in headlines]
        #print(output)
        # Combine all head patterns into a single regex pattern
        news_pattern = r"(?i)(?<!\(c\) Copyright)Copyright(?: ©)?\s+\d{4}(?:,)?\s+(.+?)(?:\nAll Rights Reserved|\nLength:|\nDistribution:|\nSection:|\n|$)|©\s+(\d{4})\s+(PCM Uitgevers B\.V\.)\s+All\s+rights\s+reserved|(Times Publishing Company)\nSection|\b(LMerc)\b|(The Xinhua General Overseas News ServiceXinhua General News Service)|(Reach PLC)"
        news_sources = re.findall(news_pattern, output, re.MULTILINE and re.IGNORECASE)
        news_sources = [list(filter(None, lst)) for lst in news_sources]
        news_sources = [reduce(lambda x, y: x + y, lst) for lst in news_sources]
        #print(output)
        # Remove 'Client/Matter: -None-' entries
        #Extra entries that do not fit
        news_sources = [instance.replace("All Rights Reserved", "").strip() for instance in news_sources]
        #The excluded instances are chosen in cases the regex pattern captured irrelevant important, add more sources here to exclude other news sources
        #Some local and state news agencies were excluded for easier generalization. For example, in the case of UPI (which focuses on more niche news sources),
        #the original news source is removed and UPI is taken as main source
        #The same was applied iwth the instances of small journals (ex. Heritage Florida Jewish News, which operates under another news agency)
        exclusions = [
        "Heritage Florida Jewish News",
        "United Feature Syndicate, Inc.",
        "The Tampa Tribune and may not be republished without permission. E-mail",
        "Content Engine, LLC.",
        "Lassen County Times",
        "Comtex News Network, Inc.",
        "by The Associated Press. .)",
        "by Qatar News Agency Distributed by UPI",
        "by Islamic Republic News Agency Distributed by UPI",
        "by Sudan News Agency Distributed by UPI",
        "by Jordan News Agency. Distributed By UPI.",
        "by United Press International.",
        "by Iran News Agency. Distributed By UPI.",
        "by Iranian News Agency Distributed by UPI",
        "ABC Radio Networks -BODY-",
        "teaches students in"
        ]

        news_sources = [match for match in news_sources if all(exclusion not in match for exclusion in exclusions)]

        print(len(headlines))
        print(headlines)
        print(len(news_sources))
        print(news_sources)
        #Creates a dataframe based on the extracted
        df =pd.DataFrame()
        df["body"]= body
        print(rawnews_dir[e])
        #removes line breaks
        df["body"].str.replace("\n", ' ')
        df['headline'] = headlines
        df["word_count"] = df['body'].str.split().str.len()
        df["news_source"]=news_sources
        df["armed_group"]=org_name[e]
        df["type"]=org_type[e]
        #removes the texts that are too large, the reason for removing such texts is that these is that longer texts might include more irrelevant topics such as news transcripts which
        #include news about multiple subjects, not limited to the interest of the research
        df= df[~(df["word_count"]>1000)]
        df_list.append(df)
    final_df=pd.concat(df_list)
    final_df.reset_index(drop=True, inplace=True)
    final_df["body"]=final_df["body"].drop_duplicates(keep="first")
    final_df.to_csv("processed_news/{}_whole.csv".format(org_name[e]))

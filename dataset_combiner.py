# importing required modules
import pandas as pd
from pypdf import PdfReader
import regex as re
from functools import reduce
import os, os.path
#Lists for organization names and their natures
org_name=["Hamas", "Hezbollah", "IRA", "LTTE"]
org_type=["Religious", "Religious", "Ethnic", "Ethnic"]
rawnews_dir=['raw_news/' + sub  for sub in org_name]
numb=0
final_df=pd.DataFrame()
#goes through the unprocessed news based on the organization
for e in range(len(rawnews_dir)):
    df_list=[]
    path=rawnews_dir[e]
    roll=len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])
    print(roll)
    #Concatenates the newspaper pages
    for a in range(roll):
        output=" "
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
        headlines=re.findall(r"\d+\.([\s\S]+?)(?=Client\/)", output, re.DOTALL)
        headlines = [s.rstrip() for s in headlines]
        head_pattern= [sub + "\n(.*)" for sub in headlines]
        #Creates a dataframe based on the extracted
        df =pd.DataFrame(columns=["id","headline","body","armed_group","type"])
        df["body"]= body
        #removes line breaks
        df["body"].str.replace("\n", ' ')
        df['headline'] = headlines
        df["word_count"] = df['body'].str.split().str.len()
        df["armed_group"]=org_name[e]
        df["type"]=org_type[e]
        #removes the texts that are too large
        df= df[~(df["word_count"]>1000)]
        df_list.append(df)
    final_df=pd.concat(df_list)
    final_df["body"]=final_df["body"].drop_duplicates(keep="first")
    final_df.to_csv("processed_news/{}_whole.csv".format(org_name[e]))

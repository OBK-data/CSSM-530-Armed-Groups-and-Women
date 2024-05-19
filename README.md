# CSSM-530-Armed-Groups-and-Women

This repository contains the data for the "An analysis of the depiction of women in news media based on armed group typology" paper written for the 2024 Spring semester course CSSM 530.

## Introduction
This project is an exploratory study of how the news depiction of women differs within the context in which armed groups with different ideologies operate. The news about religious (Hamas and Hezbollah) and ethnic (LTTE and IRA) were collected from NexisUni according to the dates these groups have operated. Please check the paper in this repository for more details about the study.

## How to replicate this study?

Firstly install dependencies of this library.

```sh
pip install -r https://raw.githubusercontent.com/OBK-data/CSSM-530-Armed-Groups-and-Women/fb96f6a3247c5c694f4dbb6d2fe7268675a49647/requirements.txt
```

To replicate this study, the python files should be ran in this order. The file **run_analysis.py** should run the files in this order; however if you want to run the files separately make sure to follow this order.
### dataset_combiner.py:
This file combines and cleans the news data of the data collected from NexisUni (Formerly known as LexisNexis) and turns them into csv files
### lda_training.py:
This file does the LDA analysis through [gensim library](https://radimrehurek.com/gensim/) and creates topics with amounts ranging from 5 to 10
### lda_metrics.py:
It calculates the metrics and puts out a graph for examination.
### lda_labels.py:
It applies the LDA model (which is six topics for each armed group typology) to label the data with six different topics
### senti_vader.py:
It runs the lexicon-based sentiment analysis model [VADER](https://github.com/cjhutto/vaderSentiment) to calculate the sentiment polarity of each news texts:
### senti_hugging_distil.py:
It runs the transformer based pre-trained model [DistilRoberta-financial-sentiment](https://huggingface.co/mr8488/distilroberta-finetuned-financial-news-sentiment-analysis) to get sentiment scores.
### senti_hugging.py
It runs the transformer based pre-trained model [SieBERT](https://huggingface.co/siebert/sentiment-roberta-large-english) to get sentiment scores.
### parser_appy_cuda.py:
It applies [Stanza's](https://stanfordnlp.github.io/stanza/usage.html) Dependency Parsing file with CUDA. Make sure to set use_gpu as False if you do not have CUDA installed.
### parser_analysis.py:
It puts out the frequency table of women both as the root of the sentence and as a dependency of other words, and female as a dependency of other words.


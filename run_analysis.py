import dataset_combiner
import lda_label
import lda_metrics
import lda_training
import parser_apply_cuda
import parser_analysis
import senti_vader
import senti_hugging_distil
import senti_hugging
import senti_vader

def multi_analysis():
    # Data cleaning
    with open('dataset_combiner.py', 'r') as f:
        exec(f.read())

    # LDA training
    with open('lda_training.py', 'r') as f:
        exec(f.read())

    #LDA metrics and graphs
    with open('lda_metrics.py', 'r') as f:
        exec(f.read())

    # LDA labelling
    with open('lda_label.py', 'r') as f:
        exec(f.read())

    #Sentiment Analysis with Vader
    with open('senti_vader.py', 'r') as f:
        exec(f.read())

    #Sentiment Analysis with Distilroberta
    with open('senti_hugging_distil.py', 'r') as f:
        exec(f.read())

    #Sentiment Analysis with SieBERT
    with open('senti_hugging.py', 'r') as f:
        exec(f.read())

    #Dependency Parsing with Stanza
    with open('parser_apply_cuda.py', 'r') as f:
        exec(f.read())

    #Results of Dependency Parsing
    with open('parser_analysis.py', 'r') as f:
        exec(f.read())

if __name__ == "__main__":
    multi_analysis()

import numpy as np
import pandas as pd

import re

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')

import string
from collections import Counter
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

punctuations = string.punctuation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["xtick.labelsize"] = 7
import seaborn as sns

df = pd.read_csv('data/qaData.csv')

def cleanText(col):
    
    texts = []
    
    for text in col:
        text = re.sub("\d+", "", text)
        text = re.sub("-", "", text)
        text = re.sub("\-", "", text)
        text = re.sub("--", "", text)
        text = text.replace('’', '').replace('\'', "").replace(".", "").replace("-", "").replace("...","")
        doc = nlp(text, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations and tok not in ("hi", "-", "thank", " -", "- ", " - ", "\-", " \-", "\- ", "--")]
        tokens = ' '.join(tokens)
        
        texts.append(tokens)
    return texts
    
def freqPlots(path, freq_dict):
    
    for tag, usage in freq_dict.items():
        fig = plt.figure(figsize=(12,6))
        fig_plt = sns.barplot(x=list(usage.keys()), y=list(usage.values()))
        fig_plt.set_xlabel("Word")
        fig_plt.set_ylabel("Normalized Frequency")
        fig_plt.set_title('Frequency Top 10 - {}'.format(tag))
        fig_plt.get_figure().savefig("{}/{}.png".format(path, tag), bbox_inches='tight') 
    return

def tfidfPlots(path, tf_dict, max_words):
    
    tfidf_dict = {}

    for doc, usage in tf_dict.items():
        tfidf_dict[doc] = {}
        total_docs = 0
        for word, value in usage.items():
            for doc2, usage2 in tf_dict.items():
                if word in usage2:
                    total_docs += 1
            tfidf_dict[doc][word] = value*np.log2(1 + len(tf_dict)/total_docs) 
            
    for tag, usage in tfidf_dict.items():
        word_values = sorted(usage.items(), key=lambda kv: -kv[1])[:max_words]
        words = [wv[0] for wv in word_values]
        values = [wv[1] for wv in word_values]
        fig = plt.figure(figsize=(12,6))
        fig_plt = sns.barplot(x=words, y=values)
        fig_plt.set_xlabel("Word")
        fig_plt.set_ylabel("TF-IDF Score")
        fig_plt.set_title('TF-IDF Top 10 - {}'.format(tag))
        fig_plt.get_figure().savefig("{}/{}.png".format(path, tag), bbox_inches='tight')
    return

def unigramPlots(path_count, path_tfidf, var, max_words):
    freq_dict = {}
    tf_dict = {}

    for j,i in df.groupby(var):
        words = ' '.join(cleanText(i["Question"])).split()
        usage = Counter(words)
        
        freq_dict[j] = {u[0]:u[1]/len(words)*100 for u in usage.most_common(max_words)}
        tf_dict[j] = {u:usage[u]/np.log2(1 + usage[u]) for u in usage}
    
    freqPlots(path_count, freq_dict)
    tfidfPlots(path_tfidf, tf_dict, max_words)
    return "Success!"

print("Visualizing Analyst Corpuses")
unigramPlots("figures/analyst_freq", "figures/analyst_tfidf", "AnalystName", 10)
print("Visualizing Tag Corpuses")
unigramPlots("figures/eTag_freq", "figures/eTag_tfidf", "EarningTag2", 10)

#################################################################################
###################################VIZ TAGS BY ANALYST###########################
#################################################################################

print("Visualizing Analyst/Tag Distributions")
for a, a_d in df.groupby("AnalystName"):
    d_p = a_d.groupby("EarningTag2").size().reset_index(name="Counts")
    fig = plt.figure(figsize=(25,6))
    fig_plt = sns.barplot(x=d_p['EarningTag2'], y=d_p['Counts']/d_p['Counts'].sum())
    fig_plt.set_xlabel("Earning Tag")
    fig_plt.set_ylabel("Count")
    fig_plt.set_title('Earning Tag Distribution - {}'.format(a))
    fig_plt.get_figure().savefig("figures/analyst_eTag/{}.png".format(a), bbox_inches='tight')
    
#################################################################################
###################################VIZ Words By Analyst and Tag##################
#################################################################################
    
def analystTagTfidfPlot():

    groups = df.groupby(['AnalystName', "EarningTag2"])

    freq_dict = {}
    tf_dict = {}

    for j,i in groups:
        words = ' '.join(cleanText(i["Question"])).split()
        usage = Counter(words)
        tf_dict[j] = {u:usage[u]/np.log2(1 + usage[u]) for u in usage}

    tfidf_dict = {}

    for doc, usage in tf_dict.items():
        tfidf_dict[doc] = {}
        total_docs = 0
        for word, value in usage.items():
            for doc2, usage2 in tf_dict.items():
                if doc2[1] == doc[1]:
                    if word in usage2:
                        total_docs += 1
            tfidf_dict[doc][word] = value*np.log2(1 + len(tf_dict)/total_docs) 


    for tag, usage in tfidf_dict.items():
        word_values = sorted(usage.items(), key=lambda kv: -kv[1])[:10]
        words = [wv[0] for wv in word_values]
        values = [wv[1] for wv in word_values]
        fig = plt.figure(figsize=(12,6))
        fig_plt = sns.barplot(x=words, y=values)
        fig_plt.set_xlabel("Word")
        fig_plt.set_ylabel("TF-IDF Score")
        fig_plt.set_title('TF-IDF Top 10 - {}-{}'.format(tag[0], tag[1]))
        fig_plt.get_figure().savefig("{}/{}_{}.png".format('figures/analyst_eTag_tfidf', tag[0],tag[1]), bbox_inches='tight')

print("Visualizing Analyst/Tag TFIDF")
analystTagTfidfPlot()
print("Finished")
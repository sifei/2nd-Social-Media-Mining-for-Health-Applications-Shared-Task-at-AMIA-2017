# Author: Sifei Han

import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import nltk
import re
import numpy as np 
from gensim import *
from gensim.models import * 
import math
from hand_pmi import *

# using NRC-Canada unigram and bigrams sentiment lexicon
senti_uni = {}
with open('Lexicon/unigrams-pmilexicon.txt','rb') as f:
    for row in f.readlines():
		row = row.split('\t')
		senti_uni[row[0]] = float(row[1])

senti_bi = {}
with open('Lexicon/bigrams-pmilexicon.txt','rb') as f:
    for row in f.readlines():
		row = row.split('\t')
		senti_bi[row[0]] = float(row[1])

# creating negation related features 	
def negated(post):
    term = ['not','no','dont','cannot']
    cnt = 0
    cnt += len(re.findall('n\'t', post.lower()))
    for t in term:
		cnt += post.lower().split().count(t)
    return cnt, cnt*1.0/(len(post.split())+1)

def negator(text):
    text = re.sub(r"(n't)\b", r" not", text)    
    return re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)),text, flags=re.IGNORECASE)

lexicon = {}
cuis = []
cui2idx = {}
type2idx = {'SIDER':0,'CHV':1,'COSTART':2,'DIEGO_Lab':3}
with open('ADR_lexicon.tsv','rb') as f:
    for line in f:
		term = line.split('\t')
		lexicon[term[1]] = [term[0],term[2].strip()]
		if term[0] not in cui2idx:
			cui2idx[term[0]] = len(cui2idx)
			cuis.append(term[0])


stopwords = {}
with open("stop-words.txt",'r') as f:
    for word in f:
		stopwords[word.strip()] = 1

# read text file
def loadData(filename):
    data = []
    with open(filename,'rb') as f:
		for row in f.readlines():
	    	data.append(row.split('\t'))
    return data

# replace user mention by special word "TARGET" and remove all stopwords.
def preprocess(text):
    temp = ''
    text_processed = []
    for line in text:
        temp = re.sub(r'@[\w]+',r'TARGET',line.lower())
        for word in temp.split():
			if word in stopwords:
				temp = temp.replace(word, ' ')
		text_processed.append(temp.decode('ascii','ignore'))
    return text_processed

def cal_norm(matrix):
    matrix_new = []
    for row in matrix:
        norm = math.sqrt(sum(i**2 for i in row))
        matrix_new.append(row/norm)
    return np.array(matrix_new)

# ngram feature 	
def ngram(text, text_Train=None):
    #cv = CountVectorizer(ngram_range=(1,3), min_df=1, max_features=10000)
    cv = TfidfVectorizer(ngram_range=(1,2),use_idf=False, min_df=1, max_features=4000)
    if text_Train == None:
		matrix = cv.fit_transform(text).toarray()
    else:
        matrix = cv.fit_transform(text_Train).toarray()
        matrix = cv.transform(text).toarray()
    matrix = cal_norm(matrix)
    return cv.get_feature_names(), matrix



def ADR_feature(tweet):
    feature = 0
    cui_array = np.zeros(len(cuis))
    type_array = np.zeros(4)
    for word in tweet.split():
		if word in lexicon:
			feature += 1
			cui_array[cui2idx[lexicon[word][0]]] += 1
			type_array[type2idx[lexicon[word][1]]] += 1
    return feature, cui_array, type_array

# averaging word2vector	value of each word in the tweet to represent whole tweet
def w2v(model, tweet):
    vec = np.array([])
    count = 1
    for word in nltk.word_tokenize(tweet):
		try:
			if len(vec) == 0:
				vec = model[word]
			else:
				vec = vec + model[word]
			count += 1
		except:
			pass
    return vec/count

# Load Pointwise mutual information(PMI) pre-computed by hand_pmi.py
hand_pmi()
PMI_scores = {}
with open('hand_make.csv','rb') as f:
    reader = csv.reader(f)
    for row in reader:
		PMI_scores[row[0]] = float(row[1])
def hand_PMI(tweet):
    score = 0.0
    for word in tweet.lower().split():
		if word in PMI_scores:
			score += PMI_scores[word]
    return score
    
def build_train(filename_text, filename_label):
    model = KeyedVectors.load_word2vec_format('../word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')
    
    tweets = []
    for row in loadData(filename_text):
		tweets.append(row[0].decode('ascii','ignore'))
    labels = []
    for row in loadData(filename_label):
		labels.append(row[0])
    toCSV = []
    all_tweets = preprocess(tweets)


    print "CEHCK"
    title = ['post','label']
    tokens, matrix = ngram(all_tweets)
    print "CHECK1"
    title += tokens
    for i in range(0,400):
		n = 'w2v'+str(i)
		title += [n]

    title += ['ADR_lexicon']
    title += cuis
    title += ['SIDER','CHV','COSTART','DIEGO_Lab']
    title +=  ['unigram_score_pos','unigram_score_neg','unigram_pos_cnt','unigram_neg_cnt','bi_score_pos','bi_score_neg','bi_pos_cnt','bi_neg_cnt','uni_score','bi_score','total_score','negate_count','negate_perc','PMI_score']
   
    toCSV.append(title)
    print len(title)
    for i in range(0,len(tweets)):
		ADR_lexicon, cuis_array, type_array = ADR_feature(all_tweets[i])
		row = [tweets[i], labels[i]] + list(matrix[i]) + list(w2v(model,all_tweets[i]))+ [ADR_lexicon] + list(cuis_array) + list(type_array)
		scores = [0.0,0.0,0,0,0.0,0.0,0,0]
		wordlist = nltk.word_tokenize(all_tweets[i])
		for word in wordlist:
			if word in senti_uni:
				if senti_uni[word] > 0:
					scores[0] += senti_uni[word]
					scores[2] += 1
				else:
					scores[1] += senti_uni[word]
					scores[3] += 1
		for j in range(0,len(wordlist)-1):
			bi_words = wordlist[j] + ' ' + wordlist[j+1]
			if word in senti_bi:
				if senti_bi[word] > 0:
					scores[4] += senti_bi[word]
					scores[6] += 1
				else:
					scores[5] += senti_bi[word]
					scores[7] += 1
		scores += [scores[0]+scores[1],scores[4]+scores[5],scores[0]+scores[1]+scores[4]+scores[5]]   
		row += scores
		negate_count, negate_perc = negated(all_tweets[i])
		row  = row +[negate_count, negate_perc]
		row = row + [hand_PMI(tweets[i])]
		toCSV.append(row)
		
    with open('task1_train_tf_PMI_full.csv','wb') as f:
		w = csv.writer(f)
		w.writerows(toCSV)
    print "Done_Train"

def build_test(filename_train, filename_dev):
    model = KeyedVectors.load_word2vec_format('../word2vec_twitter_model/word2vec_twitter_model.bin', binary=True, unicode_errors='ignore')
    tweets = []
    for row in loadData(filename_train):
		tweets.append(row[0])
    toCSV = []
    all_tweets = preprocess(tweets)

    data_dev = loadData(filename_dev)
    tweets_dev = []
    for row in data_dev:
		tweets_dev.append(row[1])
    dev_tweets = preprocess(tweets_dev)


    print "CEHCK"
    title = ['post']
    tokens, matrix = ngram(dev_tweets,all_tweets)
    print "CHECK1"
    title += tokens
    for i in range(0,400):
		n = 'w2v'+str(i)
		title += [n]

    title += ['ADR_lexicon']
    title += cuis
    title += ['SIDER','CHV','COSTART','DIEGO_Lab']
    title +=  ['unigram_score_pos','unigram_score_neg','unigram_pos_cnt','unigram_neg_cnt','bi_score_pos','bi_score_neg','bi_pos_cnt','bi_neg_cnt','uni_score','bi_score','total_score','negate_count','negate_perc','PMI_score']
   
    toCSV.append(title)
    print len(title)
    for i in range(0,len(dev_tweets)):
		ADR_lexicon, cuis_array, type_array = ADR_feature(dev_tweets[i])
		row = [tweets_dev[i]] + list(matrix[i]) + list(w2v(model,dev_tweets[i]))+ [ADR_lexicon] + list(cuis_array) + list(type_array)
		scores = [0.0,0.0,0,0,0.0,0.0,0,0]
		wordlist = nltk.word_tokenize(dev_tweets[i])
		for word in wordlist:
			if word in senti_uni:
				if senti_uni[word] > 0:
					scores[0] += senti_uni[word]
					scores[2] += 1
				else:
					scores[1] += senti_uni[word]
					scores[3] += 1
		for j in range(0,len(wordlist)-1):
			bi_words = wordlist[j] + ' ' + wordlist[j+1]
			if word in senti_bi:
				if senti_bi[word] > 0:
					scores[4] += senti_bi[word]
					scores[6] += 1
				else:
					scores[5] += senti_bi[word]
					scores[7] += 1
		scores += [scores[0]+scores[1],scores[4]+scores[5],scores[0]+scores[1]+scores[4]+scores[5]]   
		row += scores
		negate_count, negate_perc = negated(dev_tweets[i])
		row = row + [negate_count, negate_perc]
		row = row + [hand_PMI(tweets_dev[i])]
		toCSV.append(row)
    
    with open('task1_test_tf_PMI_full.csv','wb') as f:
		w = csv.writer(f)
		w.writerows(toCSV)
    print "Done_TEST"
build_train('task1_full_text.txt','task1_full_label.txt')
build_test('task1_full_text.txt','task1_test.txt')

 

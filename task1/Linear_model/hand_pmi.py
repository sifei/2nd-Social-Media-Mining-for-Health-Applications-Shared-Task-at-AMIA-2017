import math
import csv
import operator

def hand_pmi():
	tweets = ''
	class1 = ''
	class2 = ''
	pos = 0
	neg = 0

	with open('../task1_all.txt','rb') as f:
	    for row in f.readlines():
		temp = row.split('\t')
		tweets += temp[3] + ' '
		if temp[2] == '0':
		    neg += 1
		    class1 += temp[3] + ' '
		if temp[2] == '1':
		    pos += 1
		    class2 += temp[3] + ' '
	class1 = class1.lower().split()
	class2 = class2.lower().split()
	allclass = tweets.lower().split()
	wordset1 = {}
	wordset2 = {}
	wordset = {}
	for i in class1:
	    if i in wordset1:
			wordset1[i] += 1
	    else:
			wordset1[i] = 1
	for i in class2:
	    if i in wordset2:
		wordset2[i] += 1
	    else:
		wordset2[i] = 1
	for i in allclass:
	    if i in wordset:
		wordset[i] += 1
	    else:
		wordset[i] = 1

	PMI_scores = []
	for word in wordset:
	    try:
		PMI = wordset2[word]*neg*1.0/pos/wordset1[word]
		PMI_scores.append([word, math.log(PMI,2)])
	    except:
			pass


	keywords1 = {}
	for key in wordset1:
	    if key not in wordset2:
			keywords1[key] = wordset1[key]

	keywords2 = {}
	for key in wordset2:
	    if key not in wordset1:
			keywords2[key] = wordset2[key]

	keywords1 = sorted(keywords1.items(),key=operator.itemgetter(1), reverse=True)
	keywords2 = sorted(keywords2.items(),key=operator.itemgetter(1), reverse=True)

	with open('hand_make.csv','wb') as f:
	    w = csv.writer(f)
	    w.writerows(PMI_scores)

import numpy as np
import csv
from sklearn.metrics import f1_score, precision_recall_fscore_support

toCSV = []
IDs = []
y_true = []
idx = 0

with open('hlp_workshop_test_set_to_release.txt','rb') as f:
    #reader = csv.reader(f)
    for row in f.readlines():
        #IDs.append(row[1])
        tmp = row.split('\t')
        IDs.append(tmp[0].strip())
        
lr1 = np.load('probas_full\lr1.npy')
lr2 = np.load('probas_full\lr2.npy')

cnns = np.zeros((len(IDs),2))
for i in range(1,11):
    filename = 'probas_full/cnn_random3_'+str(i)+'.npy'
    temp = np.load(filename)
    cnns += temp

cnns1 = np.zeros((len(IDs),2))
for i in range(1,11):
    filename = 'probas_full/cnn_random4_'+str(i)+'.npy'
    temp = np.load(filename)
    cnns1 += temp



cnns2 = np.zeros((len(IDs),2))
for i in range(1,11):
    filename = 'probas_full/cnnattword_random3_'+str(i)+'.npy'
    temp = np.load(filename)
    cnns2 += temp

cnns3 = np.zeros((len(IDs),2))
for i in range(1,11):
    filename = 'probas_full/cnnattword_random4_'+str(i)+'.npy'
    temp = np.load(filename)
    cnns3 += temp

    
average = (lr1+cnns2)/2.0
ave_label = np.argmax(average,axis=1).tolist()
toCSV = []
for i in range(len(IDs)):
    toCSV.append([IDs[i],ave_label[i]])
with open('task1_full.csv','wb') as f:
    w = csv.writer(f)
    w.writerows(toCSV)
print "DONE"



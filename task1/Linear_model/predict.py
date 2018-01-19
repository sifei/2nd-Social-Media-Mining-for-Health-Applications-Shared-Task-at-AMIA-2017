# Author: Sifei Han
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import csv
from scipy import sparse


# load training features 
def read_train():
    return pd.read_csv('task1_train_3gram_PMI_full.csv')
# load testing features 
def read_test():
    return pd.read_csv('task1_test_3gram_PMI_full.csv
# load training features (using tfidf score for ngram)	
def read_train1():
    return pd.read_csv('task1_train_tf_PMI_full.csv')
# load testing features (using tfidf score for ngram)	
def read_test1():
    return pd.read_csv('task1_test_tf_PMI_full.csv')


def main():
    train = read_train()
    train.fillna(0, inplace=True)
    train_sample = train[:].fillna(value=0)
    feature_names = list(train_sample.columns)
 
    feature_names.remove('post')
    feature_names.remove('label')
    features = train_sample[feature_names].values
    target = train_sample['label'].values
    features = csr_matrix(features)
    X,Y = features, target


    train1 = read_train1()
    train1.fillna(0, inplace=True)
    train_sample1 = train1[:].fillna(value=0)
    feature_names1 = list(train_sample1.columns)

    feature_names1.remove('post')
    feature_names1.remove('label')

    features1 = train_sample1[feature_names1].values
    target1 = train_sample1['label'].values
    features1 = csr_matrix(features1)
    X1,Y1 = features1, target1


    test = read_test()
    test.fillna(0, inplace=True)
    test_sample = test[:].fillna(value=0)
    feature_names_test = list(test_sample.columns)

    feature_names_test.remove('post')
    features_test = test_sample[feature_names_test].values
    target_test = test_sample['label'].values
    features_test = csr_matrix(features_test)
    X_test,Y_test = features_test, target_test


    test1 = read_test1()
    test1.fillna(0, inplace=True)
    test_sample1 = test1[:].fillna(value=0)
    feature_names_test1 = list(test_sample1.columns)

    feature_names_test1.remove('post')
    features_test1 = test_sample1[feature_names_test1].values
    target_test1 = test_sample1['label'].values
    features_test1 = csr_matrix(features_test1)
    X_test1,Y_test1 = features_test1, target_test1

	  # build logistic regression classifiers
    clf1 = LogisticRegression(C=0.4, class_weight='balanced')
    clf2 = LogisticRegression(C=0.3, class_weight='balanced')
    
    clf1.fit(X,Y)
    preds1 = clf1.predict_proba(X_test)
    clf2.fit(X1,Y1)
    preds2 = clf2.predict_proba(X_test1)
	  # save prediction probabilities
    preds1.dump('probas_full/lr1.npy')
    preds2.dump('probas_full/lr2.npy')
    
main()

#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.metrics import accuracy_score
clf = GaussianNB()

start_time = time.time()
clf = clf.fit( features_train,labels_train )
end_time = time.time()
print 'training time cost:',round(end_time-start_time,3),'s'

start_time = time.time()
pred = clf.predict( features_test )
end_time = time.time()
print 'predict time cost:',round(end_time-start_time,3),'s'

acc = accuracy_score( labels_test,pred)
print 'testing accuracy is %0.3f'%acc
#########################################################



#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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
from sklearn import tree
import time
from sklearn.metrics import accuracy_score
'''
mini-project question1
使用 decision_tree/dt_author_id.py 中的初始代码，准备好决策树并将它作为分类器运行起来，设置 min_samples_split=40。
可能需要等一段时间才能开始训练。准确率是多少？
training time cost: 89.648 s
predict time cost: 0.053 s
predict accuracy is 0.977
'''
'''
mini-project2 question2
你数据中的特征数是多少？
print features_train.shape[1] 
3785
'''
'''
mini-project2 question3
进入 ../tools/email_preprocess.py，然后找到类似此处所示的一行代码： 
selector = SelectPercentile(f_classif, percentile=10)
将百分位数从 10 改为 1,现在，特征数是多少？
print features_train.shape[1] 
379
'''
'''
mini-project question4
你认为 SelectPercentile 的作用是什么？大数值的百分位数是否会在所有其他条件不变的情况下使决策树更为复杂或简单？
SelectPercentile 的作用是选取部分重要特征，即减少特征数。大数值的百分位数意味着选取更多的特征，选取更多的特征将使决策树更为复杂
'''
'''
mini-project question5
当你仅使用 1% 的可用特征（即百分位数 = 1）时，决策树的准确率是多少？
training time cost: 6.246 s
predict time cost: 0.004 s
predict accuracy is 0.967
'''
clf = tree.DecisionTreeClassifier( min_samples_split=40 )

start_time = time.time()
clf = clf.fit( features_train,labels_train )
end_time = time.time()
print 'training time cost:',round(end_time-start_time,3),'s'

start_time = time.time()
pred = clf.predict( features_test )
end_time = time.time()
print 'predict time cost:',round(end_time-start_time,3),'s'

acc = accuracy_score( labels_test,pred)
print 'predict accuracy is %0.3f'%acc

#########################################################



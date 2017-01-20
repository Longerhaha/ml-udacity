#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

''' 
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

mini-project question2
减小样本，加速训练,结果降低了准确率
training time cost: 0.147 s
There are 1046 class in Chris
predict time cost: 1.456 s
accuracy: 0.884527872582
'''
#########################################################
### your code goes here ###
from sklearn import svm
import time
from sklearn.metrics import accuracy_score
'''
clf = svm.SVC(kernel="linear")

mini-project question1
在创建分类器时使用线性内核（如果你忘记此步骤，你会发现分类器要花很长的时间来训练）。
training time cost: 208.933 s
There are 881 class in Chris
predict time cost: 22.627 s
accuracy: 0.984072810011
'''
'''clf = svm.SVC( kernel="rbf")#default kernal is rbf

mini-project question3
保留上一个测试题中的训练集代码段，以便仍在 1% 的完整训练集上进行训练。将 SVM 的内核更改为“rbf”。
training time cost: 0.154 s
There are 1540 class in Chris
predict time cost: 1.467 s
accuracy: 0.616040955631
'''
clf = svm.SVC( C=10000.0,kernel="rbf")#default kernal is rbf

'''
mini-project question4
保持训练集大小不变，并且保留上一个测试题中的 rbf 内核
C=10.0:
training time cost: 0.138 s
There are 1540 class in Chris
predict time cost: 1.52 s
accuracy: 0.616040955631
C=100.0:
training time cost: 0.147 s
There are 1540 class in Chris
predict time cost: 1.388 s
accuracy: 0.616040955631
C=1000.0
training time cost: 0.15 s
There are 1177 class in Chris
predict time cost: 1.321 s
accuracy: 0.821387940842
C=10000.0:
training time cost: 0.147 s
There are 1018 class in Chris
predict time cost: 1.175 s
accuracy: 0.892491467577
'''
'''
mini-project question5
在你为 RBF 内核优化了 C 值后，你会获得怎样的准确率？该 C 值是否对应更简单或者更复杂的决策边界？
准确率更高，决策边界更复杂，因为此时过拟合了
'''
'''
mini-project question6
你已经为 RBF 内核优化了 C，现在恢复为使用完整的训练集。
较大的训练集往往能提高算法的性能，所以（通过在大数据集上调整 C 和进行训练）我们应得到相当优化的结果。
经过优化的 SVM 的准确率是多少？
training time cost: 133.797 s
There are 877 class in Chris
predict time cost: 13.459 s
accuracy: 0.990898748578
'''
'''
mini-project question7
你的 SVM（0 或 1，分别对应 Sara 和 Chris）将测试集中的元素 10 预测为哪一类？元素 26 ？还是元素 50 ？
pred[10]=1
pred[26]=0
pred[50]=1
'''
'''
mini-project question8
 测试事件的数量超过 1700——其中多少预测在“Chris” (1) 类中？（使用 RBF 内核、C=10000. 以及完整的训练集。）
 sum(pred)=877
'''
start_time = time.time()
clf = clf.fit( features_train,labels_train )
end_time = time.time()
print 'training time cost:',round(end_time-start_time,3),'s'

start_time = time.time()
pred = clf.predict( features_test )
print 'There are %d class in Chris'%sum(pred)
#print pred[10],pred[26],pred[50]
end_time = time.time()
print 'predict time cost:',round(end_time-start_time,3),'s'

acc = accuracy_score( labels_test,pred)
print 'accuracy:',acc
#########################################################



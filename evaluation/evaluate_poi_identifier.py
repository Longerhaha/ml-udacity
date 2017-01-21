#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf = clf.fit( X_train,y_train )
pred = clf.predict( X_test )
acc = accuracy_score( y_test,pred)
print 'testing accuracy is %0.3f'%acc

#mini-project evaluation question1
#对于 POI 识别符的测试集，有多少 POI 被预测到了？4
print 'POI numbers in test set is',sum(pred)
#question2:你的测试集中共有多少人？29
print 'The test set has %d people'%(pred.shape[0])
#question3如果测试集中每个人的标识符预测为 0.0（而不是 POI），那么你的准确度为多少？0.862
import numpy as np
print "all zero's accuracy is %0.3f"%(sum(pred == np.zeros(pred.shape[0])) * 1.0 /pred.shape[0])
import numpy as np
zero_iden = np.zeros(pred.shape[0])
acc = accuracy_score(zero_iden,pred)
print 'testing accuracy(zero indentify note) is %0.3f'%acc
#question4你的标识符是否有任何真正例？ 

#从y_train 和 pred矩阵可以看出来没有一个正例(true positive)

#question5 精确率是多少？0
from sklearn.metrics import precision_score
print 'precision score is %0.3f'%precision_score(y_test,pred)

#question5 召回率是多少？0
from sklearn.metrics import recall_score
print 'recall score is %0.3f'%recall_score(y_test,pred)

#真实标签 = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
#预测值   = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

#question6
#有多少 true positive？6个
#question7
#有多少 true negative？9个
#question8
#此示例中有多少 false positives？3个
#question9
#此示例中有多少 false negatives？2个

#question10
#这个分类器的精确率是多少？6/(6+3) = 0.667 

#question10
#这个分类器的召回率是多少？6/(6+2)=0.75

'''
理解指标1
我的真正率很高，这意味着当测试数据中有__时，我能很容易地将他或她标记出来。 POI

我的识别符没有很好的 _，但是有不错的 _     精确 召回
这意味着，无论我测试集中的 POI 何时被标记，我都可以明确地知道是真实的 POI。 
另一方面，我为此付出的代价是我有时候会得到假的POI。 


我的识别符没有很好的 _，但是有不错的 _。 召回 精确
这意味着，无论我测试集中的 POI 何时被标记，我都可以明确地知道那很有可能是真实的 POI 而非虚警。 
另一方面，我为此付出的代价是我有时候会错过真实的 POI，因为我实际上不太情愿触及边界情形


这是两个世界中最好的识别符，我的 false positive 和 false negative 率均为 _，这意味着我可以可靠、准确地识别 POI。 
如果我的识别符发现了 POI，那么此人几乎可以肯定是 POI，而且如果识别符没有标记某人，那么几乎可以肯定他们不是 POI。
F1 score/low

'''















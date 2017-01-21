#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)




### it's all yours from here forward!  

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
'''
clf = DecisionTreeClassifier()
clf.fit(features,labels)
pred = clf.predict(features)
acc = accuracy_score( labels,pred)

'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)
clf = clf.fit(X_train,y_train)
pred = clf.predict( X_test )
acc = accuracy_score( y_test,pred)

print 'testing accuracy is %0.3f'%acc

'''
去掉42-46行的注释,并保证36-39行代码未被注释
你将先开始构建想象得到的最简单（未经过验证的）POI 识别符。 
本节课的初始代码 (validation/validate_poi.py) 相当直白——它的作用就是读入数据，并将数据格式化为标签和特征的列表。
创建决策树分类器（仅使用默认参数），在所有数据（你将在下一部分中修复这个问题！）上训练它，并打印出准确率。
这是一颗过拟合树，不要相信这个数字！尽管如此，准确率是多少？
testing accuracy is 0.989 

去掉36-39行的注释,并保证42-46行代码未被注释
现在，你将添加训练和测试，以便获得一个可靠的准确率数字。 
使用 sklearn.cross_validation 中的 train_test_split 验证；
将30% 的数据用于测试，并设置 random_state 参数为 42（random_state 控制哪些点进入训练集，哪些点用于测试；
将其设置为 42 意味着我们确切地知道哪些事件在哪个集中； 并且可以检查你得到的结果）。
更新后的准确率是多少？
testing accuracy is 0.724


'''
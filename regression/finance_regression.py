#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )


### list the features you want to look at--first item in the 
### list will be the "target" feature

'''mini-question1
#你尝试预测什么目标？用来预测目标的输入特征是什么？
#target:bonus       input slary

#斜率和截距是多少？
slope = 5.448140
intercept = -102360.543294

假设你是一名悟性不太高的机器学习者，你没有在测试集上进行测试，而是在你用来训练的相同数据上进行了测试，
并且用到的方法是将回归预测值与训练数据中的目标值（比如：奖金）做对比。
你找到的分数是多少？
the score on the training data is 0.045509

现在，在测试数据上计算回归的分数。测试数据的分数是多少？
the score on the testing data is -1.484992
'''
'''

我们有许多可用的财务特征，就预测个人奖金而言，其中一些特征可能比余下的特征更为强大。
例如，假设你对数据做出了思考，并且推测出“long_term_incentive”特征（为公司长期的健康发展做出贡献的雇员应该得到这份奖励）可能与奖金而非工资的关系更密切。
证明你的假设是正确的一种方式是根据长期激励回归奖金，然后看看回归是否显著高于根据工资回归奖金。根据长期奖励回归奖金—测试数据的分数是多少？
the score on the testing data is -0.592713

如果你必须预测某人的奖金并且你只有一小段相关信息，你想要知道他们的工资还是收到的长期奖励吗？
长期奖励，因为-0.592713 > -1.484992,在测试集上的表现更好
现在，我们将绘制两条回归线，一条在测试数据上拟合（有异常值），一条在训练数据上拟合（无异常值）。来看看现在的图形，有很大差别，对吧？单一的异常值会引起很大的差异。
新的回归线斜率是多少？
new regression line'coef is 0.404
'''
#features_list = ["bonus", "salary"]#mini-project question1使用
features_list = ["bonus", "long_term_incentive"]#mini-project question2使用
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"


### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.


from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit( feature_train,target_train )
print 'slope = %f'%reg.coef_
print 'intercept = %f'%reg.intercept_
print 'the score on the training data is %f'%reg.score( feature_train,target_train )
print 'the score on the testing data is %f'%reg.score( feature_test,target_test )

### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test),color='b' )
except NameError:
    pass
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()

print "new regression line'coef is %0.3f"%reg.coef_
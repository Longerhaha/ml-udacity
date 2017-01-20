#!/usr/bin/python

import random
import numpy
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner


### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )



### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit( ages_train,net_worths_train )

print 'slope = %f'%reg.coef_
print 'intercept = %f'%reg.intercept_
print 'the score on the training data is %f'%reg.score( ages_train,net_worths_train )
print 'the score on the testing data is %f'%reg.score( ages_test,net_worths_test )

'''
mini-project question 1
Sebastian 向我们描述了改善回归的一个算法，你将在此项目中实现该算法。你将在接下来的几个测试题中运用这一算法。
总的来说，你将在所有训练点上拟合回归。舍弃在实际 y 值和回归预测 y 值之间有最大误差的 10% 的点。
先开始运行初始代码 (outliers/outlier_removal_regression.py) 和可视化点。
一些异常值应该会跳出来。部署一个线性回归，其中的净值是目标，而用来进行预测的特征是人的年龄（记得在训练数据上进行训练！）。
数据点主体的正确斜率是 6.25（我们之所以知道，是因为我们使用该值来生成数据）；你的回归的斜率是多少？
slope = 5.077931
当使用回归在测试数据上进行预测时，你获得的分数是多少？
the score on the testing data is 0.878262
'''
try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()


### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
except NameError:
    print "your regression object doesn't exist, or isn't name reg"
    print "can't make predictions to use in identifying outliers"


### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        print 'After delete 10% abnormal data:'
        print 'slope = %f'%reg.coef_
        print 'intercept = %f'%reg.intercept_
        print 'the score on the training data is %f'%reg.score( ages_train,net_worths_train )
        print 'the score on the testing data is %f'%reg.score( ages_test,net_worths_test )
        plt.plot(ages, reg.predict(ages), color="blue")
        '''
        mini-project question 2
        现在当异常值被清除后，你的回归的新斜率是多少？
        After delete 10% abnormal data:
        slope = 6.368595
        当使用回归在测试集上进行预测时，新的分数是多少？
        the score on the testing data is 0.983189
        '''
    except NameError:
        print "you don't seem to have regression imported/created,"
        print "   or else your regression object isn't named reg"
        print "   either way, only draw the scatter plot of the cleaned data"
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()


else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"


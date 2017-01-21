#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

from sklearn import tree

from sklearn.metrics import accuracy_score
'''min_samples_split=40''' 
clf = tree.DecisionTreeClassifier( )

clf.fit(features_train,labels_train)


#stuck in:how to get the most important feature's name
#fixed:the quetion is What is the number of this feature(the most important feature),instead of its name,so I add the annotation below 
#fea_name = vectorizer.get_feature_names()
impor_fea = clf.feature_importances_
for index,fea in enumerate(impor_fea):
    if fea >= 0.2:
        print fea,index
train_pred = clf.predict(features_train)
test_pred = clf.predict(features_test)
train_acc = accuracy_score(labels_train,train_pred)
test_acc = accuracy_score(labels_test,test_pred)
print 'train accuracy is %0.3f'%train_acc
print 'test accuracy is %0.3f'%test_acc

'''
mini-project feature_selection
如果决策树被过拟合，你期望测试集的准确率是非常高还是相当低？
低
如果决策树被过拟合，你期望训练集的准确率是高还是低？
高

过拟合算法的一种传统方式是使用大量特征和少量训练数据。你可以在 feature_selection/find_signature.py 中找到初始代码。 
准备好决策树，开始在训练数据上进行训练，打印出准确率。
根据初始代码，有多少训练点？
print len(labels_train)   150

现在先注释掉 ../text_learning/vectorize_text.py中的57、59行代码，并运行vectorize_text.py，然后再运行本py文件
你刚才创建的决策树的准确率是多少？
0.948

选择（过拟合）决策树并使用 feature_importances 属性来获得一个列表， 其中列出了所有用到的特征的相对重要性（由于是文本数据，因此列表会很长）。
 我们建议迭代此列表并且仅在超过阈值（比如 0.2——记住，所有单词都同等重要，每个单词的重要性都低于 0.01）的情况下将特征重要性打印出来。
最重要特征的重要性是什么？该特征的数字是多少？
0.764705882353 33614
这个单词是：sshacklensf
从某种意义上说，这一单词看起来像是一个异常值，所以让我们在删除它之后重新拟合。
返回至 text_learning/vectorize_text.py，使用我们删除“sara”、“chris”等的方法，从邮件中删除此单词。
去掉vectorize_text.py 57行代码的注释
重新运行 vectorize_text.py，完成以后立即重新运行 find_signature.py。
有跳出其他任何的异常值吗？是什么单词？像是一个签名类型的单词？（跟之前一样，将异常值定义为重要性大于 0.2 的特征）。
有，cgermannsf，不像是，所以删除
去掉vectorize_text.py 59行代码的注释

再次更新 vectorize_test.py 后重新运行。然后，再次运行 find_signature.py。
是否出现其他任何的重要特征（重要性大于 0.2）？有多少？它们看起来像“签名文字”，还是更像来自邮件正文的“邮件内容文字”？
有，1个，邮件内容文字，所以保留
现在决策树的准确率是多少？
0.817

'''


















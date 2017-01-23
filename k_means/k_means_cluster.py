#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
    ### if you like, place red stars over points that are POIs (just for funsies)
        if mark_poi:
            for ii, pp in enumerate(pred):
                if poi[ii]:
                    plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
#features_list = [poi, feature_1, feature_2] #question1
features_list = [poi, feature_1, feature_2 ,feature_3]  #question2
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
#for f1, f2 in finance_features:   #question1
for f1, f2, _ in finance_features:  #question2

    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(finance_features)
pred = kmeans.predict(finance_features)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
    
'''mini-project feature scaling
哪类缩放被部署了？
salary scaling 
原始值为 20 万美元的“salary”特征和原始值为 1 百万美元的“exercised_stock_options”特征的重缩放值会是多少？
 （确保呈现浮点型而非整数型数字！）
0.180 0.029
有人可能会质疑是否必须重缩放财务数据，也许我们希望 10 万美元的工资和 4 千万美元的股票期权之间存在巨大差异。
如果我们想基于“from_messages”（从一个特定的邮箱帐号发出的电子邮件数）和“salary”来进行集群化会怎样？
 在这种情形下，特征缩放是不必要的，还是重要的？
 是重要的，因为salary比from_messages大好几个数量级，所以特征缩放是必要的
'''  
salary_ = data[:,1]
minval = numpy.min(salary_[numpy.nonzero(salary_)])
maxval = numpy.max(salary_[numpy.nonzero(salary_)])
print 'salary = '
sa_in = int(raw_input())
print 'feature scaling on salary',(sa_in-minval)/(maxval-minval)

#salary exercised_stock_options   
exercised_stock_options_ = data[:,2]
minval = numpy.min(exercised_stock_options_[numpy.nonzero(exercised_stock_options_)])
maxval = numpy.max(exercised_stock_options_[numpy.nonzero(exercised_stock_options_)])
print 'exercised_stock_options = '
exercised_stock_options_in = int(raw_input())
print 'feature scaling on exercised_stock_options',(exercised_stock_options_in-minval)/(maxval-minval)
'''
mini-project k_means question 1（去掉53、65行的注释，同时保证51、54、66行的代码是注释着的）
#首先你将基于两个财务特征开始执行 K-means，请查看代码并确定代码使用哪些特征进行聚类。
salary,exercised_stock_options
将聚类预测存储到名为 pred 的列表，以便脚本底部的 Draw() 命令正常工作。在弹出的散点图中，聚类是否是你预期的？
不是我所预期的，因为五个点成一类，其余成一类，有点不合聚类的目地。

mini-project k_means question 2（去掉51、54、66行的代码是注释，同时保证53、65的代码是注释着的）
向特征列表（features_list）中添加第三个特征：“total_payments”。现在使用 3 个，而不是 2 个输入特征重新运行聚类
（很明显，我们仍然可以只显示原来的 2 个维度）。将聚类绘图与使用 2 个输入特征获取的绘图进行比较。
是否有任何点切换群集？多少个点？这种使用 3 个牲的新聚类无法通过肉眼加以猜测——必须通过 k-均值算法才能识别它。

当你加入一些新的特征时，有测试点移动到不同的聚类中吗？
有，有四个点移动了。现在只有一个点一类，其余点一类
通过观察数据列表，“exercised_stock_options”的最大值和最小值分别是多少呢？（忽略“NaN”）
print numpy.min(exercised_stock_options_[numpy.nonzero(exercised_stock_options_)])
3285.0
print numpy.max(exercised_stock_options_[numpy.nonzero(exercised_stock_options_)])
34348384.0
“salary”取的最大值和最小值是什么？
print numpy.min(salary_[numpy.nonzero(salary_)])
477.0
print numpy.max(salary_[numpy.nonzero(salary_)])
1111258.0

特征缩放化之后，哪些数据点改变了聚类？
有两个点，具体根据题目和你运行出来的图做对比。
'''







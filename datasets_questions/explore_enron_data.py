#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
 
enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
'''
#数据集中有多少数据点（人）？
len(enron_data)
146
对于每个人，有多少个特征可用？
len(enron_data['ALLEN PHILLIP K'])
21
#E+F 数据集中有多少 POI？
num_poi = 0;
for i in enron_data:
    if enron_data[i]["poi"] == True:
        num_poi += 1
print 'the number of poi in dataset:',num_poi
the number of poi in dataset: 18


我们编辑了一个包含所有 POI 姓名的列表（在 ../final_project/poi_names.txt 中）并附上了相应的邮箱地址（在 ../final_project/poi_email_addresses.py 中）。
总共有多少 POI？(使用姓名列表，不要用邮箱地址，因为许多雇员不止一个邮箱，而且其中少数人员不是安然的雇员，我们没有他们的邮箱地址。)
poi_list_handle = open("../final_project/poi_names.txt","r");
poi_list = poi_list_handle.read()
poi = poi_list.splitlines()

#头两行是无用信息
poi.remove(poi[0])
poi.remove(poi[0])
poi_list_handle.close()
print 'poi list size is',len(poi)
poi list size is 35

James Prentice 名下的股票总值是多少？
enron_data['PRENTICE JAMES']['total_stock_value']
1095040

我们有多少来自 Wesley Colwell 的发给嫌疑人的电子邮件？
enron_data['COLWELL WESLEY']['from_this_person_to_poi']
11

Jeffrey K Skilling 行使的股票期权价值是多少？
enron_data['SKILLING JEFFREY K']['exercised_stock_options']
19250000

欺诈案发生的多数时间内，安然的 CEO 是谁？
Jeffrey Skilling/Jeffrey K Skilling
安然的董事会主席是谁？
Kenneth Lay
欺诈案发生的多数时间内，安然的 CFO（首席财务官）是谁？
Andrew Fastow

这三个人（Lay、Skilling 和 Fastow）当中，谁拿回家的钱最多（“total_payments”特征的最大值）？Kenneth Lay
这个人得到了多少钱？ 103559793
enron_data['SKILLING JEFFREY K']['total_payments']
Out[4]: 8682716
enron_data['LAY KENNETH L']['total_payments']
Out[3]: 103559793
enron_data['FASTOW ANDREW S']['total_payments']
Out[2]: 2424083

对于数据集中的所有人，不是每一个特征都有值。当特征没有明确的值时，我们使用什么来表示它？NaN

此数据集中有多少雇员有量化的工资？已知的邮箱地址是否可用？
salary_num = 0
email_address_num = 0
for i in enron_data:
    if enron_data[i]["email_address"] != 'NaN':
        email_address_num += 1
    if enron_data[i]["salary"] != 'NaN':
        salary_num += 1
print 'the number of employees with quantify salary in dataset is %d'%salary_num
print 'the number of employees with email address in dataset is %d'%email_address_num
结果：
the number of employees with quantify salary in dataset is 95
the number of employees with email address in dataset is 111

'''
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

features_list = ['total_payments']
total_payments_data = featureFormat(enron_data, features_list,remove_all_zeroes=False, sort_keys = True)
total_payments_cnt = 0
for index in total_payments_data:
    if index == 0:
        total_payments_cnt += 1
print 'There are %d people total_payments set to be NaN'%total_payments_cnt
print 'The percentage is %0.3f'%(total_payments_cnt*1.0/len(total_payments_data))
'''（当前的）E+F 数据集中有多少人的薪酬总额被设置了“NaN”？数据集中这些人的比例占多少？
result:
There are 21 people total_payments set to be NaN
The percentage is 0.144
'''


features_list = ['poi','total_payments']
data = featureFormat(enron_data, features_list, sort_keys = True)

poi,total_payments = targetFeatureSplit(data)
total_payments_cnt = 0
poi_num = 0
for index,val in enumerate(poi):
    if val==1.0:
        poi_num += 1
        if total_payments[index]==0:
            total_payments_cnt += 1
print 'There are %d people(expected to be poi) total_payments set to be NaN'%total_payments_cnt
print 'The percentage is %0.3f'%(total_payments_cnt*1.0/len(total_payments))
print "poi's number is %d"%poi_num
'''
E+F 数据集中有多少 POI 的薪酬总额被设置了“NaN”？这些 POI 占多少比例？
result:
There are 0 people(expected to be poi) total_payments set to be NaN
The percentage is 0.000
poi's number is 18
如果机器学习算法将 total_payments 用作特征，你希望它将“NaN”值关联到 POI 还是非 POI？
上一个问题中所有poi的薪酬总额没有一个被设为NaN,所以此题选非POI
'''
'''
append_data_cnt = 10
for i in range(append_data_cnt):
    poi.append([1.0])
    total_payments
    '''
'''
如果你再次添加了全是 POI 的 10 个数据点，并且对这些雇员的薪酬总额设置了“NaN”，你刚才计算的数字会发生变化。
数据集中这些人的数量变成了多少？薪酬总额被设置了“NaN”的雇员数变成了多少？
146+10 = 156,21 + 10 = 31
数据集中的 POI 数量变成了多少？股票总值被设置了“NaN”的 POI 占多少比例？
18+10=28,10/28
在添加了新的数据点后，你是否认为，监督式分类算法可将 total_payments 为“NaN”理解为某人是 POI 的线索？
'''




























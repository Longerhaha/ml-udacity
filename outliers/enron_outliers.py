#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
'''
mini-project question3
有一个异常值应该会立即跳出来。现在的问题是识别来源。该数据点的字典键名称是什么？
numpy.max(data) = 97343619.0
然后从data_dict中找到该值（bonus），发现其对应的键值是TOTAL

在此数据集上运行机器学习时，该异常值是否像我们应该包含的数据点？是否应该删除它？
应该删除它
#delete the max abnormal data


'''
data_dict.pop('TOTAL',0)
'''
写下这样的一行代码（你必须修改字典和键名）并在调用 featureFormat() 之前删除异常值。
然后重新运行代码，你的散点图就不会再有这个异常值了。
所有异常值都没了吗？
错，又出现了四个异常值。在之前的坐标看不出来这四个异常值，因为之前的那个异常值太大，导致这四个异常值没有在图上得到直观的显示
'''

data = featureFormat(data_dict, features)

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### your code below
'''我们认为还有 4 个异常值需要调查；让我们举例来看。两人获得了至少 5 百万美元的奖金，以及超过 1 百万美元的工资；换句话说，他们就像是强盗。
和这些点相关的名字是什么？
LAY KENNETH L
SKILLING JEFFREY K
你是否会猜到这些就是我们应该删除的错误或者奇怪的电子表格行，你是否知道这些点之所以不同的重要原因？
（换句话说，在我们试图构建 POI 识别符之前，是否应该删除它们？）
应该保留下来，应该这两个称呼就是人名，不是奇怪的电子表格行，所以他们是有效数据，需要保留下来
'''
for key in data_dict:
    if data_dict[key]['bonus'] >= 5000000 and data_dict[key]['bonus'] != 'NaN'and data_dict[key]['salary'] >= 1000000 and data_dict[key]['salary'] != 'NaN':
        print key


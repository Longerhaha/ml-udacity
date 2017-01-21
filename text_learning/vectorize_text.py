#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""
from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        #temp_counter += 1
        #if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            print path
            email = open(path, "r")
    
            ### use parseOutText to extract the text from the opened email
            text = parseOutText(email)
            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            text = text.replace("sara","")
            text = text.replace("shackleton","")
            text = text.replace("chris","")
            text = text.replace("germani","")
            
            #fea_name[33614] sshacklensf is an abnormal word,so delete it 
            text = text.replace("sshacklensf","")
            #fea_name[14343] cgermannsf is also an abnormal word,so continue deleting it
            text = text.replace("cgermannsf","")
            ### append the text to word_data
            word_data.append(text)
            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append([0])
            else:
                from_data.append([1])
            email.close()

print "emails processed"
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )



### in Part 4, do TfIdf vectorization here
#my code has some problem
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
#设立停止词
sw = stopwords.words('english')
TVec = TfidfVectorizer(sublinear_tf=True,analyzer='word',stop_words=sw)

#TVec = TfidfVectorizer( analyzer='word',stop_words=sw )

feature = TVec.fit_transform(word_data)
feature = feature.toarray()
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(feature)
fea = TVec.get_feature_names()


'''
41、42行代码未被注释
删除签名文字（“sara”、“shackleton”、“chris”、“germani”——如果你知道为什么是“germani”而不是“germany”，你将获得加分）
51-54行代码
向 word_data 添加更新的文本字符串——如果邮件来自 Sara，向 from_data 添加 0（零），如果是 Chris 写的邮件，则添加 1。
63-66行代码
在以下方框中，放入你得到的 word_data[152] 字符串。
tjonesnsf stephani and sam need nymex calendar
41、42行代码注释
使用 sklearn TfIdf 转换将 word_data 转换为 tf-idf 矩阵。删除英文停止词。
你可以使用 get_feature_names() 访问单词和特征数字之间的映射，该函数返回一个包含词汇表所有单词的列表。有多少不同的单词？
38865
你 TfId 中的单词编号 34597 是什么？
fea[34597] = 'statement'
上面两个答案是我自己通过代码得出来的，可是与答案不一样，更改了很多次代码还是与答案不合，欢迎指出错误。
我的微信号：cql80238023
'''



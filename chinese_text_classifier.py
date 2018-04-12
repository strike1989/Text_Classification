#coding:utf-8
import jieba
import pandas as pd
df_technology = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()

df_car = pd.read_csv("./data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()

df_entertainment = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()

df_military = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()

df_sports = pd.read_csv("./data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()

technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]

print technology[12]

print car[100]

print entertainment[10]

print military[10]

print sports[10]

stopwords=pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values

def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            segs=jieba.lcut(line)
            segs = filter(lambda x:len(x)>1, segs)
            segs = filter(lambda x:x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
        except Exception,e:
            print line
            continue

#生成训练数据
sentences = []

preprocess_text(technology, sentences, 'technology')
preprocess_text(car, sentences, 'car')
preprocess_text(entertainment, sentences, 'entertainment')
preprocess_text(military, sentences, 'military')
preprocess_text(sports, sentences, 'sports')

import random
random.shuffle(sentences)

for sentence in sentences[:10]:
    print sentence[0], sentence[1]

from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)

print len(x_train)

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    analyzer='word', # tokenise by character ngrams
    max_features=4000,  # keep the most common 1000 ngrams
)
vec.fit(x_train)

def get_features(x):
    vec.transform(x)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)

print classifier.score(vec.transform(x_test), y_test)

print len(x_test)

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    analyzer='word', # tokenise by character ngrams
    ngram_range=(1,4),  # use ngrams of size 1 and 2
    max_features=20000,  # keep the most common 1000 ngrams
)
vec.fit(x_train)

def get_features(x):
    vec.transform(x)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)
classifier.score(vec.transform(x_test), y_test)

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

def stratifiedkfold_cv(x, y, clf_class, shuffle=True, n_folds=5, **kwargs):
    stratifiedk_fold = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle)
    y_pred = y[:]
    for train_index, test_index in stratifiedk_fold:
        X_train, X_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

NB = MultinomialNB
print precision_score(y, stratifiedkfold_cv(vec.transform(x),np.array(y),NB), average='macro')

import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class TextClassifier():

    def __init__(self, classifier=MultinomialNB()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,4), max_features=20000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

text_classifier = TextClassifier()
text_classifier.fit(x_train, y_train)
print(text_classifier.predict('这 是 有史以来 最 大 的 一 次 军舰 演习'))
print(text_classifier.score(x_test, y_test))

from sklearn.svm import SVC
svm = SVC(kernel='linear')
svm.fit(vec.transform(x_train), y_train)
print svm.score(vec.transform(x_test), y_test)

# from sklearn.svm import SVC
# svm = SVC()
# svm.fit(vec.transform(x_train), y_train)
# print svm.score(vec.transform(x_test), y_test)
#
# import re
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC


class TextClassifier():

    def __init__(self, classifier=SVC(kernel='linear')):
        self.classifier = classifier
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=12000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)

text_classifier = TextClassifier()
text_classifier.fit(x_train, y_train)
print(text_classifier.predict('这 是 有史以来 最 大 的 一 次 军舰 演习'))
print(text_classifier.score(x_test, y_test))
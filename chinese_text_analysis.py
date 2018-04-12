import jieba.analyse as analyse
import pandas as pd
df = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines=df.content.values.tolist()
content = "".join(lines)
print("  ".join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=())))

from gensim import corpora, models, similarities
import gensim

stopwords=pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values

import jieba
import pandas as pd
df = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines=df.content.values.tolist()

sentences=[]
for line in lines:
    try:
        segs=jieba.lcut(line)
        segs = filter(lambda x:len(x)>1, segs)
        segs = filter(lambda x:x not in stopwords, segs)
        sentences.append(segs)
    except Exception as e:
        continue

for word in sentences[5]:
    print(word)

dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

print(corpus[5])

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)

print(lda.print_topic(3, topn=5))

for topic in lda.print_topics(num_topics=20, num_words=8):
    print(topic[1])
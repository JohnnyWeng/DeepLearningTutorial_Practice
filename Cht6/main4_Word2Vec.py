import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Packages for numerical operations and plotting
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Load the machine learning package
from sklearn.decomposition import PCA

#Load the Word2vec package
import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

import jieba
import re

f = open("TopMachineLearningPaper.txt", 'r',encoding="utf-8")
lines = []
for line in f:
    temp = jieba.lcut(line)
    words = []
    for i in temp:
        #Filter out all punctuation
        # 特殊字元轉成空字串
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        # greater than 0 will be added into words
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
print('lines[0:2] = ', lines[0:2]) # print out the first two entries in the lines list.
# Call Word2vec's algorithm for training.

model = Word2Vec(lines, vector_size= 20, window = 2 , min_count = 1)
print('similar to ai = ', model.wv.most_similar('AI', topn =5))

# ------------------------------------------------------------------------------------

rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.index_to_key):
    rawWordVec.append(model.wv[w])
    word2ind[w] = i
print('word2ind = ', word2ind)
rawWordVec = np.array(rawWordVec)
print('rawWordVec.shape = ',rawWordVec.shape)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)
print('X_reduced.shape = ',X_reduced.shape)

fig = plt.figure(figsize = (30, 20))
# fig.gca() retrieves the "current" axes on the figure fig, creating a plotting area.
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.3, color = 'black')

words = ['machine', 'learning', 'Algorithms', 'papers', 'AI', 'ChatGPT', 'language', 'analysis', 'Time', 'Models', 'Voice', 'improve', 'deep']
zhfont1 = matplotlib.font_manager.FontProperties(fname='./chinese_song.ttf', size=16)
logging.info("Font loaded successfully.")
# Loop through the 13 words and see if they
for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        print('xy = ', xy) # The x and y axis of the words.
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'green')
        plt.text(xy[0], xy[1], w, alpha = 1, color = 'blue')
logging.info(" [End] ")
# Show the plot
plt.show()
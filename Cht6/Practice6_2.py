# Load necessary packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import jieba
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and process data
f = open("TopMachineLearningPaper.txt", 'r', encoding="utf-8")
lines = []
for line in f:
    temp = jieba.lcut(line)
    words = []
    for i in temp:
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
f.close()

# Train Word2Vec model
model = Word2Vec(lines, vector_size=20, window=2, min_count=1)

# Define the list of words
words = ['machine', 'learning', 'Algorithms', 'papers', 'AI', 'ChatGPT', 'language', 'analysis', 'Time', 'Models', 'Voice', 'improve', 'deep']

# Prepare word vectors for plotting
rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.index_to_key):
    rawWordVec.append(model.wv[w])
    word2ind[w] = i
rawWordVec = np.array(rawWordVec)
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)

# Calculate distances from "AI" to each word in the list
distances = {}
# Calculate the distance between "AI"
ai_index = word2ind['AI']
ai_coords = X_reduced[ai_index]  # Coordinates of "AI" in the reduced space

for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]

        # Calculate Euclidean distance from "AI"
        distance = np.linalg.norm(xy - ai_coords)
        distances[w] = distance

sorted_words = sorted(distances, key=distances.get)[:6]  # Includes "AI" itself
print('sorted_words = ', sorted_words)
# Draw the 2D plot
fig = plt.figure(figsize=(30, 20))
ax = fig.gca()
ax.set_facecolor('white')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.3, color='black')

# Plot "AI" and its top 5 closest words
zhfont1 = matplotlib.font_manager.FontProperties(fname='./chinese_song.ttf', size=16)
for w in sorted_words:
    ind = word2ind[w]
    xy = X_reduced[ind]
    plt.plot(xy[0], xy[1], '.', alpha=1, color='green')
    plt.text(xy[0], xy[1], w, alpha=1, color='blue', fontproperties=zhfont1)

logging.info(" [End] ")
plt.show()

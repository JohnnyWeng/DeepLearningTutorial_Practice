
import jieba

#Load the package for regular expression
import re

#Read in the original file
f = open("TopMachineLearningPaper.txt", 'r',encoding="utf-8")
text = str(f.read())
f.close()

# split the words
temp = jieba.lcut(text)
print('temp[0:20] = ', temp[0:20])

words = []
for i in temp:
    #Filter out all punctuation
    # All the thing inside the [] will be replaced with "". We want to get rid of the space.
    i = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、~@#￥%……&*（）：]+", "", i)
    if len(i) > 0:
        words.append(i)
print(len(words))
print(words[0:20]) # print out the first 20 data.

import re
def main():
    content = 'abc124hello46goodbye67'
    list1 = re.findall(r'\d+', content)
    print('list1 = ', list1)
    mylist = list(map(int, list1))
    print(mylist)
    print('sum(mylist) = ', sum(mylist)) #237
    # This line replaces any sequence of digits followed by either the letter 'h' or 'g' in the content string with the string 'foo1'.
    print(re.sub(r'\d+[hg]', 'foo1', content))
    print(re.sub(r'\d+', '456654', content))
main()
i=0

trigrams = [([words[i], words[i + 1]], words[i + 2]) for i in range(len(words) - 2)]
# Print out the first three elements
print(trigrams[:3])

vocab = set(words)
print(len(vocab))
word_to_idx = {}
idx_to_word = {}
ids = 0

#For the full-text loop, construct these two dictionaries
for w in words:
    cnt = word_to_idx.get(w, [ids, 0])
    if cnt[1] == 0:
        idx_to_word[ids] = w
        ids += 1
    cnt[1] += 1
    word_to_idx[w] = cnt

print(word_to_idx)
print("-"*100)
print(idx_to_word)

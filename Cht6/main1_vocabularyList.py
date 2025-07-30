import collections

# Stop word setting
stop_words=['\n', 'or', 'are', 'they', 'i', 'some', 'by', '—',
            'even', 'the', 'to', 'a', 'and', 'of', 'in', 'on', 'for',
            'that', 'with', 'is', 'as', 'could', 'its', 'this', 'other',
            'an', 'have', 'more', 'at','dont', 'can', 'only', 'most']

# A collection of vocabulary lists
word_freqs = collections.Counter()
word_freqs["aaa"] += 1;
with open('news.txt','r+', encoding='UTF-8') as f:
    for line in f:
        words = line.lower().replace('.', ' ').split()
        # Filter and Count the number of word occurrences
        for word in words:
            if not (word in stop_words):
                word_freqs[word] += 1
print(word_freqs.most_common(20)) # list 最多的前 20 筆

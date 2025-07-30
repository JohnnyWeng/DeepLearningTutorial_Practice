import nltk
nltk.download('punkt')  # punkt tokenizer
nltk.download('punkt_tab')


# Article
text = "Today is a great day. It is even better than yesterday." + \
       " And yesterday was the best day ever."

# Split paragraphs: 切句點
print('split to sentences = ',nltk.sent_tokenize(text))  # split to sentences
print('\n')
# Split words: 切空白
print('split to words = ', nltk.word_tokenize(text))  # split to words

text = "In Brazil they drive on the right-hand side of the road. Brazil has a large coastline on the eastern side of South America"
print('\n')
from nltk.tokenize import word_tokenize
token = word_tokenize(text)
print('token= ',token)
print('\n')
from nltk.probability import FreqDist
fdist = FreqDist(token)
print('fdist= ', fdist) # total: 24 words, 扣掉重複的剩下 18 個 samples
print('\n')
#The 10 most frequent words: 印出出現幾次
fdist1 = fdist.most_common(10)
print('fdist1= ', fdist1)

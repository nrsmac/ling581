# imports
import string
import random
import nltk
from collections import defaultdict, Counter
from nltk.corpus import cess_esp, stopwords
from nltk import FreqDist, ngrams
from tqdm import tqdm
 

sents = cess_esp.sents()
 
# write the removal characters such as : Stopwords and punctuation
stop_words = set(stopwords.words('spanish'))
string.punctuation = string.punctuation +'"'+'"'+'-'+'''+'''+'—'
removal_list = list(stop_words) + list(string.punctuation)+ ['lt','rt']
 
# generate unigrams bigrams trigrams
unigram=[]
bigram=[]
trigram=[]
tokenized_text=[]
for sentence in tqdm(sents, desc="removing stopwords and making unigrams"):
  sentence = list(map(lambda x:x.lower(),sentence))
  for word in sentence:
        if word== '.':
            sentence.remove(word)
        else:
            unigram.append(word)
   
  tokenized_text.append(sentence)
  bigram.extend(list(ngrams(sentence, 2,pad_left=True, pad_right=True)))
  trigram.extend(list(ngrams(sentence, 3, pad_left=True, pad_right=True)))
 
def remove_stopwords(x):    
    y = []
    for pair in x:
        count = 0
        for word in pair:
            if word in removal_list:
                count = count or 0
            else:
                count = count or 1
        if (count==1):
            y.append(pair)
    return (y)
unigram = remove_stopwords(unigram)
bigram = remove_stopwords(bigram)
trigram = remove_stopwords(trigram)
 
freq_bi = FreqDist(bigram)
freq_tri = FreqDist(trigram)
 
d = defaultdict(Counter)
for a, b, c in tqdm(freq_tri, desc="Counting trigram frequencies"):
    if(a != None and b!= None and c!= None):
      d[a, b][c] += freq_tri[a, b, c]
       
 
def pick_word(counter):
    "Chooses a random element."
    return random.choice(list(counter.elements()))

while True:
    s=input("Introduce dos palabras para generar\n>")
    prefix = tuple(s.split())
    s = " ".join(prefix)
    try:
        for i in range(19):
            suffix = pick_word(d[prefix])
            s=s+' '+suffix
            prefix = prefix[1], suffix
        print(s)
    except Exception:
        print("Invalid input")

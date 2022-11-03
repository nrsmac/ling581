import spacy
import sys
from spacy import displacy

with open(sys.argv[1]) as f:
    text = f.read()

nlp=spacy.load('en_core_web_sm')
doc = nlp(text)
for sent in doc.sents:
    print(f"{sent}")
    for tok in sent:
        print(f"{tok.text}: {tok.dep_} {tok.head}")
    print("\n")

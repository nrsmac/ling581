import spacy
from spacy import displacy

SOURCE = './kapital.txt'
OUTPUT = f'./{SOURCE[:-4]}_pos.txt'

with open(SOURCE,"r") as f:
    lines = f.readlines()
    text = ' '.join(lines)

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

outlines = ["TEXT\tLEMMA\tPOS"]
for token in doc:
    outlines.append(f"{token.text}\t{token.lemma_}\t{token.pos_}\n")

with open(OUTPUT, "w") as f:
    f.writelines(outlines)

    
displacy.serve(doc, "dep", options={"compact":True, "distance":100, "collapse_punct":True})

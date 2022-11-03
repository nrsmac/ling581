import sys

if len(sys.argv) != 2:
    print("USAGE: {sys.argv[0]} [INPUT FILE]")
    exit(0)

from transformers import pipeline

docs_file = sys.argv[1]
with open(docs_file, "r") as f:
    docs = f.read().splitlines()

sentiment_pipeline = pipeline('sentiment-analysis')

out = ""
sentiments = sentiment_pipeline(docs)
for i, sentiment_dict in enumerate(sentiments):
    label, score = sentiment_dict['label'], sentiment_dict['score']
    out+=f"TEXT:\n {docs[i]}\n{label=}{score=}\n\n"

with open(f"{docs_file.split('.')[0]}_out.txt", "w") as f:
    f.write(out)

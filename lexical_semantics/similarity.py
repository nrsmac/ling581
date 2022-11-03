import spacy 
from tqdm import tqdm

nlp = spacy.load("en_core_web_lg")

# Discovering geopolitical perception and bias in word embeddings 

with open("./countries.txt","r") as f:
    countries = f.read().splitlines()

keywords = ["happy","free", "rich", "poor", "religious", "secular", "educated", "charitable", "prosperous", "welfare", "economy", "trade", "war", "slavery", "industry"]

def rank_countries(keyword_doc:nlp):
   rankings = [(keyword_doc.similarity(nlp(c)), c) for c in countries]
   rankings.sort(reverse=True)
   return rankings

countries_by_keyword = {} 

for keyword in (pbar := tqdm(keywords)):
    keyword_doc = nlp(keyword)
    countries_by_keyword[keyword] = rank_countries(keyword_doc)

with open("results.txt", "w") as f:
    for i, keyword in enumerate(keywords):
        f.write(f'Top 10 "{keyword}" countries are:\n')
        for rank, country in countries_by_keyword[keyword][:10]:
            f.write(f'{country}:{rank}\n')
        
        f.write(f'\nLeast 10 "{keyword}" countries are:\n')
        for rank, country in list(reversed(countries_by_keyword[keyword]))[:10]:
            f.write(f'{country}:{rank}\n')
        f.write("\n")

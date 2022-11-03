import nltk
from nltk.parse.generate import generate, demo_grammar
from nltk import CFG
grammar = CFG.fromstring(f"""
    S -> NP VP
    NP -> Det N
    NP -> Det Adj N
    NP -> Det N PP
    NP -> Det Adj N PP
    NP -> N PP
    PP -> P NP
    VP -> V | V NP | V PP | V NP PP | Adv V NP | V N Adv | 
    Det -> 'the' | 'a'
    Adj -> 'exuberant' | 'dreadful'
    V -> 'slept' | 'saw' | 'ate' | 'drank'
    N -> 'man' | 'park' | 'dog' 
    P -> 'in' | 'with' | 'on' 
        """)
print(demo_grammar)

for sentence in generate(grammar, depth=5):
    print(' '.join(sentence))

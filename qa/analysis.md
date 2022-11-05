For this assignment, I decided to play with RoBERTa for question answering. I built a Python script to run an NLP pipeline. 

To run for yourself:
`python roberta.py -c CONTEXT_FILE.txt -q QUESTION_FILE.txt`
If no question is supplied, it will run in interactive mode. 

Three different texts were chosen in three different genres:
* An excerpt from the novel __Never Let Me Go__ by Kashuo Ishiguro
* A news article (about the Elon Musk twitter acquisition 
* An encyclopedia article about my favourite marine animal, the nautilus

I asked a wide array of questions about each document to test the model's abilities. For the most part, it produced accurate answers to most reasonable questions I asked it. The answers tend to be fairly extractive, and I would like questions answered in full sentences or at least in a more original/expressive manner.

You will also see many duplicate questions which is intended to test both how consistent the model is and how well it deals with ambiguities. For example, with the novel document as context:  
* __What special privilege does she have?__ -> she gets to pick and chose 
* __What special privilege does Kathy have?__ -> a great record 

* __Where is she from?__ -> Dover
* __Where is Kathy from?__ -> Hillsham

I also probed it's summarizing capabilities in each context:
* (Novel) __What is this about?__ -> Carers aren't machines (This is interesting as it's a primary theme of the novel, but probably this sentence by chance).
* (News Article) __What is this about?__ -> the folks that had incited violence (incorrect, could may be correct by some stretch, but I would give it half points)
* (News Article) __What is this article about?__ -> election-related misinformation (still wrong)
* (Encylopedia Entry) __What is this about?__ -> the secret to how the nautilus swims (not quite)
* (Encylopedia Entry) __What is this text about?__ chambered or pearly nautilus (yay!)

My favourite result is one from the nautilus article. I asked: __Which phylum does the nautilus belong to?__, to which it correctly answered: cephalopod. While this word is in the document, "phylum" isn't. I believe this is because in the word embedding, cephalopod must be semantically close to phylum. 

I think a further direction for research would be on improving the output of these models, which I'm sure has been attempted. Maybe redirecting output from RoBERTa into a decoder-only architecture like GPT-3, or simply fine-tuning somehow could remedy this. 

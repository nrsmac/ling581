import argparse
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('-f','--file', required=True)
args = vars(parser.parse_args())

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

with open(args['file'], 'r') as f:
    context = f.read()

print(f"Context [{args['file']}]: {context}")
while True:
    QA_input = {'question': input('Question: '), 'context': context}
    res = nlp(QA_input)
    print(f"{res['answer']}")


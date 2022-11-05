import argparse
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('-c','--context_file', required=True)
parser.add_argument('-q','--questions_file', required=False)
args = vars(parser.parse_args())

model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

with open(args['context_file'], 'r') as f:
    context = f.read()

if args['questions_file']:
    with open(args['questions_file'], 'r') as f:
        questions = f.readlines()

    with open(f"{args['context_file'][:-4]}_answers.txt", 'w') as f:
        for question in tqdm(questions):
            QA_input = {'question': question, 'context': context}
            answer = nlp(QA_input)['answer']
            f.write(f"{answer}\n")
            

else:
    print(f"Context [{args['context_file']}]: {context}")

    while True:
        QA_input = {'question': input('Question: '), 'context': context}
        res = nlp(QA_input)
        print(f"{res['answer']}")


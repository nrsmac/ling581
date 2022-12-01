from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

model_name = './results_ja2zh/checkpoint-500'
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

prefix = 'translate Spanish to Japanese: '
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

while True:
	src_text = f"{prefix}{input('Enter a phrase in English: ')}"
	translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
	print(f'Spanish: {[tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]}')


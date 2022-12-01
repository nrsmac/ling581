from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

model_name = './results_en2es/checkpoint-59500'
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

prefix = 'translate English to Spanish: '

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

while True:
	src_text = f"{prefix}{input('Enter a phrase in English: ')}"
	translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
	print(f'Spanish: {[tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]}')

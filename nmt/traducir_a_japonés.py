from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

#news_commentary = load_dataset('news_commentary', 'en-es')
#news_commentary = news_commentary['train'].train_test_split(test_size=0.2, seed=42)

model_name = './results_es2ja/checkpoint-500'
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

#source_lang = 'es'
#target_lang = 'ja'
prefix = 'translate Spanish to Japanese: '

#def preprocess_function(examples):
#    inputs = [prefix + example[source_lang] for example in examples["translation"]]
#    targets = [example[target_lang] for example in examples["translation"]]
#    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
#    return model_inputs

#print(f"tokenizing {len(news_commentary['train'])}")
#tokenized_news_commentary = news_commentary.map(preprocess_function, batched=True)
#data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

while True:
	src_text = f"{prefix}{input('Enter a phrase in English: ')}"
	translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
	print(f'Spanish: {[tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]}')


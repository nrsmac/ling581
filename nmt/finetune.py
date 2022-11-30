from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

books = load_dataset('news_commentary', 'es-ja')
books = books['train'].train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained('t5-base')

source_lang = 'es'
target_lang = 'ja'
prefix = 'translate Spanish to Japanese: '

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

tokenized_books = books.map(preprocess_function, batched=True)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        fp16=False,
        report_to="wandb"
)

trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_books["train"],
        eval_dataset=tokenized_books["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
)

trainer.train()
model.save_pretrained('./results/model')

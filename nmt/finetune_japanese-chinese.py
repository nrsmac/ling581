from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

news_commentary = load_dataset('news_commentary', 'ja-zh')
news_commentary = news_commentary['train'].train_test_split(test_size=0.2, seed=42)

tokenizer = AutoTokenizer.from_pretrained('t5-small')

source_lang = 'ja'
target_lang = 'zh'
prefix = '日本語を中国語に翻訳して: '

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

print(f"tokenizing {len(news_commentary['train'])}")
tokenized_news_commentary = news_commentary.map(preprocess_function, batched=True)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
        output_dir="./results_ja2zh",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=100,
        fp16=True,
        report_to="wandb"
)

trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_news_commentary["train"],
        eval_dataset=tokenized_news_commentary["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
)

trainer.train()

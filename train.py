from datasets import load_dataset
from transformers import GPT2TokenizerFast,Trainer,GPT2LMHeadModel,TrainingArguments

# 定义译码器
block_size = 128
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为填充符号
# 定义模型和数据集
model = GPT2LMHeadModel.from_pretrained('gpt2')
dataset = load_dataset('text',data_files={'train': './data/train.txt','test':'./data/test.txt'})
# 定义编码函数
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=block_size)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenizer_dataset = dataset.map(tokenize_function, batched=True)
print(tokenizer_dataset)
# 定义训练超参数
training_args = TrainingArguments(
    output_dir='./data',
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10000,
    save_total_limit=2,
)
# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenizer_dataset['train'],
    # eval_dataset=tokenizer_dataset['test'],
)
# 训练并保存模型
trainer.train()
trainer.save_model("./data/model")



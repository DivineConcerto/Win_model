from transformers import GPT2LMHeadModel,GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('./data/model')

def generate_text(input_text):
    input_ids = tokenizer.encode(input_text,return_tensors='pt')
    output = model.generate(input_ids,max_length=50,temperature=0.7)
    generated_text = tokenizer.decode(output[0],skip_special_tokens=True)
    print(generated_text)

if __name__ == '__main__':
    while True:
        text = input("请输入内容:")
        generate_text(text)
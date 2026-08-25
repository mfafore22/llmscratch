import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

encode = tokenizer.encode
decode = tokenizer.decode

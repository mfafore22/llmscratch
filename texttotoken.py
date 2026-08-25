import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


if __name__ == "__main__":
    import tiktoken
    from generatetext import generate_text_simple
    from gpttextgeneration import model
    from gptconfig import GPT_CONFIG_124M

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding('gpt2')

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("allenai/llama")
    model = AutoModelForCausalLM.from_pretrained("allenai/llama")

    while True:
        input_text = input("User: ")
        if input_text.lower() == "exit":
            break

        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Response: ", response)


if __name__ == "__main__":
    main()

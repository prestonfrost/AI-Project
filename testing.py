from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load the tokenizer and your fine-tuned model from the saved checkpoint
checkpoint_path = "./results/checkpoint-10000"  
tokenizer = MBart50Tokenizer.from_pretrained(checkpoint_path, src_lang="en_XX", tgt_lang="es_XX")
model = MBartForConditionalGeneration.from_pretrained(checkpoint_path)

while True:
    # Take input sentence from the user
    input_sentence = input("Enter an English sentence (or type 'quit' to exit): ")
    if input_sentence.lower() == "quit":
        break

    # Prepend the English language token
    input_sentence = f"<en_XX> {input_sentence}"

    # Tokenize the input sentence
    inputs = tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Generate the translation using the fine-tuned model
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"],  # Force the output to start in Spanish
        max_length=128,
        num_beams=8,  # Use beam search to improve translation quality
        early_stopping=True
    )

    # Decode the generated tokens into a Spanish sentence
    translated_sentence = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print("Translated to Spanish:", translated_sentence)

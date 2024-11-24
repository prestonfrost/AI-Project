from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50Tokenizer, Trainer, TrainingArguments

# Load dataset in streaming mode
dataset = load_dataset("opus100", "en-es", split='train', streaming=True)

# Load the correct tokenizer and model for MBart50
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

# Tokenization function for both directions
def tokenize_function(examples):
    inputs = []
    targets = []

    for example in examples["translation"]:
        if example["en"] and example["es"]:
            # English to Spanish
            inputs.append(f"<en_XX> {example['en']}")
            targets.append(example["es"])

            # Spanish to English
            inputs.append(f"<es_XX> {example['es']}")
            targets.append(example["en"])

    # Tokenize inputs and targets
    tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    tokenized_targets = tokenizer(targets, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    # Create a dictionary to store model inputs
    model_inputs = {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_targets["input_ids"],
    }

    return model_inputs

# Tokenize the dataset in smaller batches
tokenized_datasets = dataset.map(tokenize_function, batched=True, batch_size=100, remove_columns=["translation"])

# Define training arguments, including max_steps
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",  # Disable evaluation
    save_steps=1000,
    logging_dir='./logs',
    logging_steps=100,
    per_device_train_batch_size=8,  # Adjust batch size according to memory limits
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # Adjust if memory allows
    num_train_epochs=1,
    load_best_model_at_end=False,
    max_steps=10000,  # Specify the maximum number of training steps
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,  # Pass tokenized dataset for training
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

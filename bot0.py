import firebase_admin
from firebase_admin import credentials, firestore
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
import warnings
from flask import Flask, request, jsonify

app = Flask(__name__)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Initialize Firebase Admin SDK
cred = credentials.Certificate("acc.json")  # Replace with your service account key file
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load tokenizer and model for DistilGPT2
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

@app.route('/bot', methods=['POST'])
def chat_bot():
    user_input = request.json.get('message', '')
    bot_response = generate_response(user_input)
    return jsonify({'response': bot_response})

# Function to load conversation history from Firestore
def load_history():
    history_ref = db.collection("conversation_history").document("history")
    history_doc = history_ref.get()
    if history_doc.exists:
        return history_doc.to_dict().get("history", [])
    else:
        return []

# Function to save conversation history to Firestore
def save_history(history):
    history_ref = db.collection("conversation_history").document("history")
    current_history = load_history()
    current_history.extend(history)
    history_ref.set({"history": current_history})

# Fine-tune the DistilGPT2 model on your dataset
def fine_tune_gpt2(dataset_path):
    # Load and tokenize your dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=128  # Adjust block size as needed
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine-tuned-distilgpt2",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
    )

    # Define data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Create Trainer and fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()

# Main function to run the bot
def main():
    # Load conversation history
    conversation_history = load_history()

    # Fine-tune DistilGPT2 on your dataset
    fine_tune_gpt2("dataset.txt")
    
    # Function to generate bot responses
    def generate_response(input_text, max_new_tokens=50):
        # Tokenize the input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True)
            
        # Generate the bot response
        outputs = model.generate(
            input_ids,
            max_length=max_new_tokens,
            do_sample=True,  # Enable sampling
            top_k=50,
            temperature=0.7,  # Adjust temperature here
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode the generated response
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return bot_response

    def process_input(user_input):
        nonlocal conversation_history  # Access the conversation history from the outer scope
        # Generate bot response
        bot_response = generate_response(user_input)
        
        # Reset conversation history
        conversation_history.clear()

        return bot_response

    while True:
        user_input = input("You: ")
        bot_response = process_input(user_input)
        print("Bot:", bot_response)

        # Update conversation history
        conversation_history.append({"user_input": user_input, "bot_response": bot_response})
        save_history(conversation_history)

if __name__ == '__main__':
    main()

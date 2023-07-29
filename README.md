# TeacherChatBot
Generating Teacher responses

# Teacher Bot
This is a ChatBot that generates Teacher Responses. The ChatBot is trained using the Train dataset from the BEA shared task 2023.
Teacher Bot is a simple chatbot that uses TF-IDF and cosine similarity to generate responses to user input. It also includes a basic greeting mechanism.

## Getting Started

### Prerequisites

To run the Teacher Bot, you will need Python 3.x and the required Python packages listed in `requirements.txt`.
## Requirements.txt
nltk
scikit-learn
bert-score

Interact with the chatbot by typing your messages. Type "bye" to exit the conversation.

### Usage

1. Place your training data in the `train_with-reference.jsonl` file.

2. Run the `teacher_bot.py` script:


## How it Works

The chatbot uses TF-IDF vectorization and cosine similarity to find the most similar response to the user's input from the training data. It also has a basic greeting mechanism to respond to greetings.

## Evaluation

The chatbot's responses can be evaluated using BERTScore. We have included an example of how to use BERTScore for evaluation in the code.

## Acknowledgments

This chatbot was inspired by various tutorials and examples available online.


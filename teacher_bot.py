# 
text_data = []
with open('train_with-reference.jsonl', 'r') as f:
    for line in f:
        text_data.append(line.strip())
import random
import string
import warnings

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

nltk.download('popular', quiet=True)

lemmer = WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


sent_tokens = []
sent_tokens.extend(text_data)

print("Teacher Bot: Hello! I am your teacher. How can I help you today?")

while True:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            print("Teacher Bot: You are welcome..")
            break
        else:
            if greeting(user_response) is not None:
                print("Teacher Bot: " + greeting(user_response))
            else:
                print("Teacher Bot: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        print("Teacher Bot: Bye! Have a great day.")
        break

pip install bert_score

import bert_score

# Reference sentences
ref_sentences = ["Hello, how are you?", "I am fine, thank you."]

# Generated sentences
gen_sentences = ["Hi, how are you doing?", "I am doing well, thank you."]

# Calculate BertScore
P, R, F1 = bert_score.score(gen_sentences, ref_sentences, lang='en', verbose=True)

# Print results
print("Precision:", P.mean().item())
print("Recall:", R.mean().item())
print("F1 score:", F1.mean().item())
print("Accuracy:",)

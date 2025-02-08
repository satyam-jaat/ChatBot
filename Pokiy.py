import json
import spacy
import random
import speech_recognition as sr
from gtts import gTTS
from collections import deque
from google.colab import drive, output
from IPython.display import display, Audio
import time

# Mount Google Drive
drive.mount('/content/drive')

# Load English language model for spaCy
nlp = spacy.load("en_core_web_md")

# Path to the existing data.json file in Drive
data_path = "/content/drive/My Drive/data.json"

class ChatBot:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.question_queue = deque(maxlen=5)
        self.mode = "question_asking"
        self.similarity_threshold = 0.75

    def load_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    # def get_similarity(self, text1, text2):
    #     doc1 = nlp(text1.lower())
    #     doc2 = nlp(text2.lower())
    #     return doc1.similarity(doc2)
    
    def get_similarity(self, text1, text2):
      # Process the texts
      doc1 = nlp(text1.lower())
      doc2 = nlp(text2.lower())
    
      # Create sets of lemmatized tokens, filtering out stopwords and non-alphabetic tokens
      tokens1 = {token.lemma_ for token in doc1 if not token.is_stop and token.is_alpha}
      tokens2 = {token.lemma_ for token in doc2 if not token.is_stop and token.is_alpha}
    
      # If one of the token sets is empty, return 0 similarity
      if not tokens1 or not tokens2:
        return 0.0
    
      # Compute the Jaccard similarity (intersection over union)
      intersection = tokens1.intersection(tokens2)
      union = tokens1.union(tokens2)
      similarity = len(intersection) / len(union)
    
      return similarity



    # def speak(self, text):
    #     """Convert text to speech and play it before continuing."""
    #     tts = gTTS(text=text, lang='en')
    #     audio_path = "/content/response.mp3"
    #     tts.save(audio_path)
    #     display(Audio(audio_path, autoplay=True))
    #     time.sleep(len(text) / 5)  # Wait based on text length to ensure full playback

    def speak(self, text):
        """Convert text to speech and play it before continuing."""
        tts = gTTS(text=text, lang='en')
        audio_path = "/content/response.mp3"
        tts.save(audio_path)
        display(Audio(audio_path, autoplay=True))
        # Estimate wait time based on word count instead of character count.
        # Average speaking rate is about 2.5 words per second.
        word_count = len(text.split())
        wait_time = word_count / 2.5
        time.sleep(wait_time)


    def find_best_match(self, user_input, category):
        best_score = 0
        best_question = None
        for question in self.data[category]:
            all_variants = [question["question"]] + question["variations"]
            for variant in all_variants:
                score = self.get_similarity(user_input, variant)
                if score > best_score:
                    best_score = score
                    best_question = question
        return best_question, best_score

    def handle_general_conversation(self, user_input):
        response = None
        best_score = 0
        for q in self.data["general"]:
            all_phrases = [q["question"]] + q["variations"]
            for phrase in all_phrases:
                score = self.get_similarity(user_input, phrase)
                if score > best_score and score > self.similarity_threshold:
                    best_score = score
                    response = random.choice(q["answers"])
        return response

    def user_asking_mode(self, user_input):
        general_response = self.handle_general_conversation(user_input)
        if general_response:
            return general_response
        
        best_question, score = self.find_best_match(user_input, "technical")
        if score > self.similarity_threshold:
            return random.choice(best_question["answers"])
        else:
            return "I don't have information about that question."

    def question_asking_mode(self):
        available_questions = [q for q in self.data["technical"] if q["question"] not in self.question_queue]
        if not available_questions:
            self.question_queue.clear()
            available_questions = self.data["technical"]
        
        current_question = random.choice(available_questions)
        self.question_queue.append(current_question["question"])
        return current_question

    def evaluate_answer(self, user_answer, correct_answers):
        best_score = 0
        best_match = ""
        for correct_answer in correct_answers:
            score = self.get_similarity(user_answer, correct_answer)
            if score > best_score:
                best_score = score
                best_match = correct_answer
        return best_score * 100, best_match

    def switch_mode(self):
        self.mode = "answer_asking" if self.mode == "question_asking" else "question_asking"
        return f"Switched to {self.mode} mode."

# Initialize chatbot with existing data.json from Drive
bot = ChatBot(data_path)

# Main interaction loop for Google Colab
print("Chatbot: Hello! Type 'exit' to end or 'switch mode' to change modes.")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    if user_input.lower() == 'switch mode':
        print(bot.switch_mode())
        continue
    
    if bot.mode == "question_asking":
        question_data = bot.question_asking_mode()
        print(f"Chatbot: {question_data['question']}")
        bot.speak(question_data['question'])
        user_answer = input("Your answer: ").strip()
        
        if user_answer.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        if user_answer.lower() == 'switch mode':
            print(bot.switch_mode())
            continue
        
        score, best_match = bot.evaluate_answer(user_answer, question_data["answers"])
        print(f"Chatbot: Your answer matched {score:.1f}% with our records.")
        if score < 80:
            print(f"Suggested answer: {best_match}")
    else:  # Answer asking mode
        response = bot.user_asking_mode(user_input)
        print(f"Chatbot: {response}")
        bot.speak(response)

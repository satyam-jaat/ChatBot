import json
import spacy
import random
import speech_recognition as sr
from gtts import gTTS
from collections import deque
from google.colab import drive
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
        self.mode = "general_mode"  # Default mode
        self.similarity_threshold = 0.75

        self.total_questions_asked = 0
        self.total_questions_answered = 0
        self.total_skipped = 0


    def load_data(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    # ------------------- SIMILARITY CHECKS -------------------
    def get_input_similarity(self, user_input, stored_inputs):
        """Calculate similarity for general inputs."""
        doc1 = nlp(user_input.lower())

        best_score = 0
        for stored_input in stored_inputs:
            doc2 = nlp(stored_input.lower())
            score = doc1.similarity(doc2)
            best_score = max(best_score, score)

        return best_score

    def get_question_similarity(self, user_question, stored_question):
        """Calculate similarity for technical questions."""
        doc1 = nlp(user_question.lower())
        doc2 = nlp(stored_question.lower())
        return doc1.similarity(doc2)

    # ------------------- GENERAL CONVERSATION -------------------
    def handle_general_conversation(self, user_input):
        """Handle general inputs and return appropriate responses."""
        best_response = None
        best_score = 0

        for entry in self.data["general"]:
            score = self.get_input_similarity(user_input, entry["inputs"])
            if score > best_score and score > self.similarity_threshold:
                best_score = score
                best_response = random.choice(entry["responses"])

        return best_response if best_response else "I'm not sure about that."

    # ------------------- TECHNICAL QUESTIONS -------------------
    def ask_technical_question(self):
        """Ask a technical question from the dataset and track the count."""
        available_questions = [q for q in self.data["technical"] if q["question"] not in self.question_queue]

        if not available_questions:
            self.question_queue.clear()
            available_questions = self.data["technical"]

        question_data = random.choice(available_questions)
        self.question_queue.append(question_data["question"])
        
        self.total_questions_asked += 1  # Track total questions asked

        return question_data


    def evaluate_technical_answer(self, user_answer, correct_answers):
        """Evaluate user's answer based on similarity with correct answers."""
        best_score = 0
        best_match = ""

        for correct_answer in correct_answers:
            score = self.get_question_similarity(user_answer, correct_answer)
            if score > best_score:
                best_score = score
                best_match = correct_answer

        return best_score * 100, best_match

    def handle_technical_question(self, user_input):
        """Find and return the best-matching technical question's answer."""
        best_question, score = self.find_best_technical_match(user_input)
        if best_question and score > self.similarity_threshold:
            return random.choice(best_question["answers"])
        else:
            return "I don't have information about that question."

    def find_best_technical_match(self, user_input):
        """Find the best technical question match."""
        best_score = 0
        best_question = None

        for question in self.data["technical"]:
            score = self.get_question_similarity(user_input, question["question"])
            if score > best_score:
                best_score = score
                best_question = question

        return best_question, best_score

    # ------------------- MODE SWITCHING -------------------
    def switch_mode(self):
        """Toggle between Technical and General modes."""
        self.mode = "technical_mode" if self.mode == "general_mode" else "general_mode"
        return f"Switched to {self.mode.replace('_', ' ')} mode."

    # ------------------- SPEECH OUTPUT -------------------
    def speak(self, text):
        """Convert text to speech and play it."""
        tts = gTTS(text=text, lang='en')
        audio_path = "/content/response.mp3"
        tts.save(audio_path)
        display(Audio(audio_path, autoplay=True))

        # Estimate wait time based on word count (2.5 words per second)
        time.sleep(len(text.split()) / 2.5)

# Initialize chatbot with data.json
bot = ChatBot(data_path)

# ------------------- MAIN CHAT LOOP -------------------
print("Chatbot: Hello! Type 'exit' to end or 'switch mode' to change modes.")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    
    if user_input.lower() == 'switch mode':
        print(bot.switch_mode())
        continue

    if bot.mode == "general_mode":
        response = bot.handle_general_conversation(user_input)
        print(f"Chatbot: {response}")
        bot.speak(response)

 # Technical Mode: Ask and evaluate technical questions
    else:  
        question_data = bot.ask_technical_question()
        print(f"Chatbot: {question_data['question']}")
        bot.speak(question_data['question'])
        
        user_answer = input("Your answer (or type 'skip' to pass): ").strip()

        if user_answer.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        if user_answer.lower() == 'switch mode':
            print(bot.switch_mode())
            continue
        if user_answer.lower() in ['skip', 'pass']:  # Allow skipping
            bot.total_skipped += 1  # Track skipped questions
            print("Chatbot: Question skipped!")
            continue

        score, best_match = bot.evaluate_technical_answer(user_answer, question_data["answers"])
        
        if score > 0:
            bot.total_questions_answered += 1  # Track answered questions

        print(f"Chatbot: Your answer matched {score:.1f}% with our records.")
        
        if score < 80:
            print(f"Suggested answer: {best_match}")

            
# Show performance summary at the end
print("\n===== Chatbot Performance Summary =====")
print(f"Total Questions Asked: {bot.total_questions_asked}")
print(f"Total Questions Answered: {bot.total_questions_answered}")
print(f"Total Questions Skipped: {bot.total_skipped}")

# Calculate performance based on answered questions only
if bot.total_questions_asked - bot.total_skipped > 0:
    performance_score = (bot.total_questions_answered / (bot.total_questions_asked - bot.total_skipped)) * 100
    print(f"Overall Performance: {performance_score:.1f}%")
else:
    print("Overall Performance: 0% (No questions answered)")

import spacy
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChatBotLogic:
    def __init__(self, training_files):
        self.nlp = spacy.load("en_core_web_sm")
        self.qa_pairs = []

        for file_path in training_files:
            if os.path.exists(file_path):
                self.load_qa_from_file(file_path)

        self.questions = [self.nlp(q) for q, _ in self.qa_pairs]

    def load_qa_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    q, a = line.strip().split('|', 1)
                    self.qa_pairs.append((q.strip(), a.strip()))

    def get_response(self, message):
        message_doc = self.nlp(message)
        if not self.questions:
            return "Sorry, I have no knowledge yet.", 0.0

        sims = [message_doc.similarity(q_doc) for q_doc in self.questions]
        best_match_idx = int(np.argmax(sims))
        best_score = sims[best_match_idx]

        if best_score < 0.6:
            return "I'm not sure I understand. Can you rephrase?", best_score

        return self.qa_pairs[best_match_idx][1], best_score


from flask import Flask, request, jsonify
from chatbot import ChatBotLogic
import os

app = Flask(__name__)

# Path to training data
training_files = ["data/eng.txt", "data/arabic.txt"]

# Initialize chatbot logic
chatbot = ChatBotLogic(training_files)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid request, 'message' field is required."}), 400
    
    message = data['message']
    response, confidence = chatbot.get_response(message)
    return jsonify({"response": response, "confidence": confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6123, debug=True)

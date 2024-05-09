from flask import Flask, render_template, request, jsonify
import pickle
import os
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    clf_loaded = pickle.load(file)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    prediction = clf_loaded.predict([email_text])[0]
    return jsonify({'email': email_text, 'prediction': 'spam' if prediction == 1 else 'not spam'})

if __name__ == "__main__":
    app.run(host="localhost", port=int(os.environ.get("PORT", 8000)))

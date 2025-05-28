from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_placement():
    try:
        # Get form inputs
        cgpa = float(request.form.get('cgpa'))
        iq = int(request.form.get('iq'))
        profile_score = int(request.form.get('profile_score'))

        # Predict using the model
        prediction = model.predict(np.array([[cgpa, iq, profile_score]]))

        result = 'placed' if prediction[0] == 1 else 'not placed'
        return render_template('index.html', result=result)
    
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    # For development use only
    app.run(host='0.0.0.0', port=8080)

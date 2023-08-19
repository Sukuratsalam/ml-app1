import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        answer = "Yes: The patient has had a heart disease attack in the past"
    else:
        answer = "No: The patient does not have a heart disease attack in the past"
   
    return render_template('index.html', prediction_text=  answer)
   

if __name__ == "__main__":
    app.run(debug=True)

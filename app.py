from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn
import numpy as np
from sklearn import preprocessing

app=Flask(__name__)
model=pickle.load(open('MentalTracker.pkl','rb'))
@app.route('/', methods=['GET'])
def home():
    return render_template('new_questionnaire.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        converted_data = {}
        for field, value in request.form.items():
            converted_data[field] = (value)

        df = pd.DataFrame(converted_data, index=[0])

        for feature in df:
            le = preprocessing.LabelEncoder()
            le.fit(df[feature])
            df[feature] = le.transform(df[feature])

        prediction = model.predict(df)[0]
        if prediction == 0:
            return render_template('result.html', prediction_text='NO NEED')
        else:
            return render_template('result.html', prediction_text="YES NEED")

if __name__=="__main__":
    app.run()
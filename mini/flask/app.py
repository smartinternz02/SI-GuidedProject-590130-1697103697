from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

app = Flask(__name__,template_folder='templates')


# Load your machine learning model
loaded_model = joblib.load('your_model.sav')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    yummy = request.form.get('yummy')
    convenient = request.form.get('convenient')
    spicy = request.form.get('spicy')
    fattening = request.form.get('fattening')
    greasy = request.form.get('greasy')
    fast = request.form.get('fast')
    cheap = request.form.get('cheap')
    tasty = request.form.get('tasty')
    expensive = request.form.get('expensive')
    healthy = request.form.get('healthy')
    disgusting = request.form.get('disgusting')
    visit_frequency = request.form.get('visit_frequency')

    # Process user inputs
    x_input = pd.DataFrame({
        '1: yummy': [1 if yummy == 'yes' else 0],
        '2: convenient': [1 if convenient == 'yes' else 0],
        '3: spicy': [1 if spicy == 'yes' else 0],
        '4: fattening': [1 if fattening == 'yes' else 0],
        '5: greasy': [1 if greasy == 'yes' else 0],
        '6: fast': [1 if fast == 'yes' else 0],
        '7: cheap': [1 if cheap == 'yes' else 0],
        '8: tasty': [1 if tasty == 'yes' else 0],
        '9: expensive': [1 if expensive == 'yes' else 0],
        '10: healthy': [1 if healthy == 'yes' else 0],
        '11: disgusting': [1 if disgusting == 'yes' else 0],
        '12: Age': [age],
        '13: VisitFrequency': [0 if visit_frequency in ['Every three months', 'Once a month'] else 2],
        '14: Gender': [1 if gender == 'Male' else 0]
    })

    

    # Make predictions using the loaded model
    prediction = loaded_model.predict(x_input)

    return render_template('index.html', prediction_result=f"Predicted class: {prediction[0]}")

if __name__ == '__main__':
    app.run(debug=True)

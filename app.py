from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("insurance_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    # Extract form values
    age = int(request.form['age'])
    sex = request.form['sex']
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    region = request.form['region']

    # Build dataframe for prediction
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])

    prediction = model.predict(input_data)
    cost = round(prediction[0], 2)

    return render_template("index.html", prediction_text=f"Predicted Insurance Cost: ${cost}")

if __name__ == "__main__":
    app.run(debug=True)

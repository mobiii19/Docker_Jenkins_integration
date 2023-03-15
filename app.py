from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('trained_salary_pred_LinearReg_model.joblib')


@app.route('/')
def index():
    return render_template('index.html', prediction=None)


@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    years_exp = request.form['years_of_experience']
    salary = model.predict(np.array([float(years_exp)]).reshape(-1, 1))[0]

    return render_template('index.html', prediction=int(salary), years_of_experience=years_exp)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("3000"), debug=True)

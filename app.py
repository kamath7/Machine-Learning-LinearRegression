from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    prediction_text = ''
    if len(list(request.form.values())) == 0:
        prediction_text = 'You need to enter a value to get a salary!'
    else:
        experience = [float(x) for x in request.form.values()]
        preds = [np.array(experience)]
        op = round(model.predict(preds)[0], 2)
        prediction_text = 'You will be receiving a salary of ${0} with {1} year(s) of experience'.format(
            op, experience[0])
    return render_template('index.html', prediction=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)

import numpy as np
import pandas as pd
import urllib
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

dataset = pd.read_csv('diabetes.csv')

# dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

dataset_X = dataset.iloc[:,[0,1,2,3,4,5,6,7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/pres')
def pres():
    return render_template('pres.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    
    if prediction == 1:
        pred = "You have Diabetes, please read the given Prescription or consult a Doctor."
        btn = "Prescription"
    elif prediction == 0:
        pred = "You don't have Diabetes. You are safe"
    output = pred

    if prediction == 1:
        return render_template('index.html', prediction_text='{}'.format(output),prescription_text = '{}'.format(btn))
    else:
        return render_template('index.html', prediction_text='{}'.format(output))
# @app.route('/make-request')
# def make_request():
#     link = request.args.get('index.html')
#     if link:
#         urllib.urlopen(link) # for Python 3.x use urllib.request.urlopen

#     # now redirect back to the referrer
#     # if no referrer, redirect to some_view
#     return redirect(request.referrer or url_for('some_view'))    

if __name__ == "__main__":
    app.run(debug=True)

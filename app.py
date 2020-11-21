import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle

# load model
svm = pickle.load(open('svm.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # get data
    data = [str(x) for x in request.form.values()]
    new_vectors = vectorizer.transform(data)
    new_predictions = svm.predict(new_vectors)
    
    print(new_predictions)
    print(type(new_predictions))
    output = {'new_predictions': new_predictions.tolist()}
    for i,row in enumerate(output):
     output[i] = {k:("Clickbait" if 1 else "Not Clickbait") for k,v in row.items()}

    # return data
    # return jsonify(results=output)

    # data = [str(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)


    return render_template('index.html', prediction_text='Results Are:  $ {}'.format(output))

if __name__ == '__main__':
    app.run(port = 5000, debug=True)

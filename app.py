import pandas as pd
from flask import Flask, jsonify, request
import pickle

# load model
svm = pickle.load(open('svm.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)
    
    # data.update((x, [y]) for x, y in data.items())
    # data_df = pd.DataFrame.from_dict(data)

    # #converts df to list
    # data_df.values.tolist()

    new_vectors = vectorizer.transform(data)
    new_predictions = svm.predict(new_vectors)
    
    # # convert data into dataframe
    # data.update((x, [y]) for x, y in data.items())
    # data_df = pd.DataFrame.from_dict(data)

    # # predictions
    # result = model.predict(data_df)

    # send back to browser
    print(new_predictions)
    print(type(new_predictions))
    output = {'new_predictions': new_predictions.tolist()}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
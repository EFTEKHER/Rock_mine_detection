from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# Load the model
with open('model/logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data', [])
    predefined = request.json.get('predefined', '')

    if predefined == 'input_data1':
        input_data =(0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032)
    elif predefined == 'input_data2':
        input_data = (0.0414,0.0436,0.0447,0.0844,0.0419,0.1215,0.2002,0.1516,0.0818,0.1975,0.2309,0.3025,0.3938,0.5050,0.5872,0.6610,0.7417,0.8006,0.8456,0.7939,0.8804,0.8384,0.7852,0.8479,0.7434,0.6433,0.5514,0.3519,0.3168,0.3346,0.2056,0.1032,0.3168,0.4040,0.4282,0.4538,0.3704,0.3741,0.3839,0.3494,0.4380,0.4265,0.2854,0.2808,0.2395,0.0369,0.0805,0.0541,0.0177,0.0065,0.0222,0.0045,0.0136,0.0113,0.0053,0.0165,0.0141,0.0077,0.0246,0.0198)
    else:
        # Pad custom input data to 60 elements
        input_data = data + [0] * (60 - len(data))

    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_as_numpy_array)
    result = "Mine" if prediction[0] == 0 else "Rock"
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)

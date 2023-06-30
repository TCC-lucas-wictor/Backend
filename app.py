from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

@app.route('/predictRandonForest', methods=['POST'])
def predictRandom():
    # Load data
    base = pd.read_csv("database_lambda.csv")
    x = base.drop("tempo", axis=1)
    y = base["tempo"]

    # Create a new instance of the model
    regression = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True)
    regression.fit(x, y)
    
    # Get prediction parameters from the request
    data = request.get_json(force=True)
    complexidade = data['complexidade']
    qtd_inputs = data['qtd_inputs']
    req_api = data['req_api']

    # Create a new dataframe from the input parameters
    mydict = {'complexidade': complexidade, 'qtd_inputs': qtd_inputs, 'req_api': req_api}
    mydict = pd.DataFrame(mydict, index=[0])

    # Predict
    y_pred = regression.predict(mydict)
    y_pred = y_pred.tolist()

    # Return result
    return jsonify(y_pred)

@app.route('/predictDecisionTree', methods=['POST'])
def predictDecision():
    # Load data
    base = pd.read_csv("database_lambda.csv")
    x = base.drop("tempo", axis=1)
    y = base["tempo"]

    # Create a new instance of the model
    regression = DecisionTreeRegressor(max_features=2)
    regression.fit(x, y)
    
    # Get prediction parameters from the request
    data = request.get_json(force=True)
    complexidade = data['complexidade']
    qtd_inputs = data['qtd_inputs']
    req_api = data['req_api']

    # Create a new dataframe from the input parameters
    mydict = {'complexidade': complexidade, 'qtd_inputs': qtd_inputs, 'req_api': req_api}
    mydict = pd.DataFrame(mydict, index=[0])

    # Predict
    y_pred = regression.predict(mydict)
    y_pred = y_pred.tolist()

    # Return result
    return jsonify(y_pred)

@app.route('/predictPolynomialRegression', methods=['POST'])
def predictPoly():
    # Load data
    base = pd.read_csv("database_lambda.csv")
    x = base.drop("tempo", axis=1)
    y = base["tempo"]

    # Create a new instance of the model
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(x)
    x_train_poly = poly.transform(x)

    # Get prediction parameters from the request
    data = request.get_json(force=True)
    complexidade = data['complexidade']
    qtd_inputs = data['qtd_inputs']
    req_api = data['req_api']

    # Create a new dataframe from the input parameters
    mydict = {'complexidade': complexidade, 'qtd_inputs': qtd_inputs, 'req_api': req_api}
    mydict = pd.DataFrame(mydict, index=[0])
    x_test_poly = poly.transform(mydict)

    # Predict
    reg = LinearRegression().fit(x_train_poly, y)
    y_pred = np.maximum(reg.predict(x_test_poly), 0)
    
    y_pred = y_pred.tolist()

    # Return result
    return jsonify(y_pred)



if __name__ == '__main__':
    CORS(app)
    app.run(port=5000, debug=True)
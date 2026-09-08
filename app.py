from flask import Flask,request,jsonify
import joblib
import numpy as np

#model building
model=joblib.load("iris_model.pkl")

app=Flask(__name__)
@app.route("/")
def home():
    return "Iris classification step"

@app.route("/predict",methods=["POST"])
def predict():
    try:
        data=request.get_json(force=True)
        if "features" not in data:
            return jsonify({"Error:col not exisy"}),400
        features=np.array(data["features"],dtype=float).reshape(1,4)

        prediction=model.predict(features)[0]
        classes=["setosa","versicolor","virginica"]
        result={"prediction":classes["prediction"]}
        return jsonify(result)
    except:
        return jsonify({"Error"}),400

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000)
    






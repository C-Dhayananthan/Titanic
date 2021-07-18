from flask import Flask,render_template,request
import os
import joblib as jb

app = Flask(__name__,template_folder="C:\PYTHON\Machine Learning\Titanic\\templates")


@app.route("/",methods = ['GET'])
def home():
    return render_template("deadprediction.html")

@app.route("/predict",methods = ["POST"])
def predict():
    if request.method == "POST":
        pclass = float(request.form.get("p-class"))
        age = float(request.form.get("Age"))
        fare = float(request.form.get("Fare"))
        gender = float(request.form.get("Gender"))
        loaded_model = jb.load("C:\PYTHON\Machine Learning\Titanic\env\death_model2")
        prediction = loaded_model.predict_proba([[pclass,age,fare,gender]])[0][1]
        return render_template("predict.html",prediction = round(100*prediction))
    else :
        return render_template('predict.html')

if __name__ == "__main__" :
    app.run(debug = True)



    
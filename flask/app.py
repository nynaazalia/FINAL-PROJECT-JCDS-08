from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import folium
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dataset",methods=["POST"])
def dataset():
    return render_template("dataset.html")

@app.route("/visualization",  methods=["POST"])
def visualization():
    return render_template("visualization.html")

@app.route("/prediction",  methods=["POST"])
def prediction():
    return render_template("prediction.html")

@app.route("/clustering",  methods=["POST"])
def clustering():
    return render_template("clustering.html")

@app.route("/result", methods=["POST","GET"])
def result():
    if request.method == "POST":
        input = request.form
        numofproduct = int(input["prod"])
        age = int(input["age"])
        tenure = int(input["tenure"])
        creditS = int(input["credit"])
        salary = float(input["salary"])
        balance = float(input["balance"])
        country = input["Country"]
        inputCountry = ""
        if country == "France":
            france = 1
            germany = 0
            spain = 0
            inputCountry = "France"
        if country == "Germany":
            france = 0
            germany = 1
            spain = 0
            inputCountry = "Germany"
        if country == "Spain":
            france = 0
            germany = 0
            spain = 1
            inputCountry = "Spain"
        gender = input["Gender"]
        inputGender = ""
        if gender == "male":
            male = 1
            inputGender = "Male"
        if gender == "female":
            male = 0
            inputGender = "Female"
        active = input["Active Member"]
        inputAct = ""
        if active == "Yes":
            activemember = 1
            inputAct = "Yes"
        else:
            activemember = 0
            inputAct = "No"
        datainput = [[creditS, age, tenure, balance, numofproduct, activemember, salary,france, spain, germany, male]]
        pred = model.predict(datainput)[0]
        proba = model.predict_proba(datainput)[0]
        if pred == 0:
            probability = round((proba[0]*100), 1)
            result = "stay in Standard Chartered Bank"
            solution = "It's a good news, let's do some strategies to improve customer loyalty!"
        else:
            probability = round((proba[1]*100), 1)
            result = "leave Standard Chartered Bank"
            solution = "It's a bad news, let's find why and do some strategies to make them stay!"
        return render_template(
            "result.html", Country= inputCountry, Gender= inputGender,age= age, tenure= tenure, credit = creditS, balance= balance,product= numofproduct, active= inputAct, salary= salary,result= result, proba = probability , solution=solution)


if __name__ == '__main__':
    model = joblib.load('modelJoblib')
    app.run(debug=True)


    
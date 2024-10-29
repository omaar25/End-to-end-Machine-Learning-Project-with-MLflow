from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.ML.pipeline.prediction import PredictionPipeline


app = Flask(__name__) 

@app.route('/',methods=['GET'])  
def homePage():
    return render_template("index.html")

@app.route('/train',methods=['GET'])  
def training():
    os.system("python main.py")
    return "Training Successful!" 

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            type =float(request.form['Type'])
            Air_temperature_C =float(request.form['Air_temperature_C'])
            Process_temperature_C =float(request.form['Process_temperature_C'])
            Rotational_Speed_RPM =float(request.form['Rotational_Speed_RPM'])
            Torque_Nm =float(request.form['Torque_Nm'])
            Tool_Wear_min =float(request.form['Tool_Wear_min'])
         
            data = [type,Air_temperature_C,Process_temperature_C,Rotational_Speed_RPM,Torque_Nm,Tool_Wear_min]
            data = np.array(data).reshape(1, 11)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('index.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')



if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)
      
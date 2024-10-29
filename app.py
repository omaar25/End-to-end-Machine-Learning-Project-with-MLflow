from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.ML.pipeline.prediction import PredictionPipeline


app = Flask(__name__) 

@app.route('/',methods=['GET'])  
def homePage():
    return render_template("index.html")

@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)
      
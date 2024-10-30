from flask import Flask, render_template, request
from ML.pipeline.AppPipeline import AppPipelinePrediction



pipeline = AppPipelinePrediction()
app = Flask(__name__)


@app.route('/', methods=['GET'])  
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            prediction = pipeline.predict(request.form)
            return render_template('index.html', prediction=prediction)
        except Exception as e:
            raise e

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

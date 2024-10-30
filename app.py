from flask import Flask, render_template, request
import os 
import numpy as np
from src.ML.pipeline.prediction import PredictionPipeline
from src.ML.components.data_processing import DataProcessing
from src.ML.config.configuration import ConfigurationManager
import pandas as pd
import pickle
app = Flask(__name__)

# Configuration setup
config = ConfigurationManager().get_data_processing_config()
processor = DataProcessing(config)

@app.route('/', methods=['GET'])  
def homePage():
    return render_template("index.html")


#@app.route('/train', methods=['GET'])  

#def training():
 #   try:
  #      os.system("python main.py")  # This should trigger your training script
   #     return "Training Successful!" 
    #except Exception as e:
     #   return f"An error occurred while training: {str(e)}"

@app.route('/predict', methods=['POST'])
def index():
    if request.method == 'POST':
        try:

            Type = request.form['Type']
            Air_temperature_K = float(request.form['Air_temperature_K'])
            Process_temperature_K = float(request.form['Process_temperature_K'])
            Rotational_Speed_RPM = float(request.form['Rotational_Speed_RPM'])
            Torque_Nm = float(request.form['Torque_Nm'])
            Tool_Wear_min = float(request.form['Tool_Wear_min'])
            data = np.array([Type, Air_temperature_K, Process_temperature_K, 
                             Rotational_Speed_RPM, Torque_Nm, Tool_Wear_min]).reshape(1,-1)
            
            columns = ['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]',
                       'Torque [Nm]','Tool wear [min]',
                       ]
            
            input_df = pd.DataFrame(data, columns=columns)


            input_df['Air temperature [K]'] = input_df['Air temperature [K]'].astype(float)
            input_df['Process temperature [K]'] = input_df['Air temperature [K]'].astype(float)
            input_df['Rotational speed [rpm]'] = input_df['Rotational speed [rpm]'].astype(float)
            input_df['Torque [Nm]'] = input_df['Torque [Nm]'].astype(float)
            input_df['Tool wear [min]'] = input_df['Tool wear [min]'].astype(float)

            processor.df = input_df
            processor.rename_columns()
            processor.convert_temperature()
            scaler_path = os.path.join(config.root_dir, "scaler.pkl")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            col_to_scale = ['Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_min', 'Air_temperature_C', 'Process_temperature_C']
            processor.df[col_to_scale] = scaler.transform(processor.df[col_to_scale])
            processor.encode_features()
            
            obj = PredictionPipeline()
            encoded_predict = obj.predict(processor.df.values)
            
            failure_type = processor.decode_failure_type(encoded_predict[0])
            
            return render_template('index.html', prediction=failure_type)

        except ValueError as ve:
            return f'Input error: {str(ve)}. Please check your inputs.'
        except Exception as e:
            print('Exception:', e)
            return 'Something went wrong during prediction.'

    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

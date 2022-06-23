# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
from utils import Utils
import pickle, logging
import numpy as np
import sklearn
import pandas as pd

app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            prediction = 0
            # reading the inputs given by the user
            gre_score = float(request.form['gre_score'])
            toefl_score = float(request.form['toefl_score'])
            university_rating = float(request.form['university_rating'])
            sop = float(request.form['sop'])
            lor = float(request.form['lor'])
            cgpa = float(request.form['cgpa'])
            is_research = request.form['research']
            if is_research == 'yes':
                research = 1
            else:
                research = 0
            logging.info(f"gre_score={gre_score}, toefl_score={toefl_score}, university_rating={university_rating}")
            logging.info(f"sop={sop}, lor={lor}, cgpa={cgpa}, research={research}")
            filename = 'admission_lr.model'
            logging.info("Model Loading")
            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            logging.info("Model Loaded")
            # predictions using the loaded model file
            logging.info("Scaler model loading")
            scaler = pickle.load(open("scaler.model", 'rb'))
            logging.info("Scaler model loaded")
            data = [gre_score, toefl_score, university_rating, sop, lor, cgpa, research]
            logging.info(data)
            data = scaler.fit_transform(pd.DataFrame(data).T)
            logging.info(data)
            prediction = loaded_model.predict(data)
            prediction = round(100 * prediction[0])
            logging.info(f"prediction={prediction}")
            # showing the prediction results in a UI
            logging.info("Render Result")
            return render_template('results.html', prediction=prediction)
        except Exception as e:
            logging.error(f"Exception={e}")
            return e
            # return render_template('results.html')
if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    Utils.init_logs()
    app.run(debug=True)  # running the app

from flask import Flask, render_template, request
from utils.preprocess import extract_data_from_image
from utils.predict import make_prediction
import os
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                if filename.lower().endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = extract_data_from_image(filepath)

                prediction_result = make_prediction(df)
            except Exception as e:
                prediction_result = f"Prediction error: {e}"

    return render_template('index.html', result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)

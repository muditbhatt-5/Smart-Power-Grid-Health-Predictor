# Smart Power Grid Health Predictor

## Features
- Upload image of a transformer record table
- Extract data using OCR (Tesseract)
- Predict faults and anomalies using a trained model
- Interactive UI with mouse hover highlights

## Steps to Run
1. Place at least 8–10 lakh rows of transformer data in `data/power_grid_sample.csv`.
2. Run the model trainer:
   ```bash
   python utils/train_model.py
   ```
3. Start the app:
   ```bash
   python app.py
   ```
4. Open browser at http://localhost:5000 and upload a table image.

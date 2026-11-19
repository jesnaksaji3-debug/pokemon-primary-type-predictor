# Pokemon Primary Type Predictor

A small Flask application that predicts a Pokemon's primary type from HP,
Attack, Defense, Height, Weight, and Base Experience. It trains a Random Forest
model from `G2.csv` on first prediction and saves it as
`pokemon_type_model.joblib` for subsequent requests.

## Run locally

From this folder, install the dependencies and start the Flask development
server:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Keep `G2.csv` beside `app.py`; the
model is trained automatically the first time a prediction is submitted.
Base Experience may be left blank, in which case the model's training median
is used.

The model uses only the six values exposed in the form. Predictions are
estimates and may be inaccurate, especially for less common types.

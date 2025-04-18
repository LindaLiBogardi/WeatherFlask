from flask import Flask, render_template
from fetch_and_predict import fetch_data_from_smhi, preprocess_data, prepare_lagged_features, \
    load_scalers_and_model, prepare_features_for_prediction, make_prediction

app = Flask(__name__)

@app.route('/')
def home():
    # Run prediction pipeline
    station_id = 66110  # Södra Öland
    df_months = fetch_data_from_smhi(station_id)
    df_daily = preprocess_data(df_months)
    df_daily = prepare_lagged_features(df_daily)
    scaler_X, scaler_rain_amt, scaler_wind, model = load_scalers_and_model()
    X_recent = prepare_features_for_prediction(df_daily, n_steps=5)
    rain_prediction, wind_velocity = make_prediction(X_recent, model, scaler_rain_amt, scaler_wind)

    return render_template('index.html', rain=rain_prediction, wind=wind_velocity)

if __name__ == '__main__':
    app.run(debug=True)

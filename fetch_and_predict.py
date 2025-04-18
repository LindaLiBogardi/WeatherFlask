
import requests
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# === 1. Fetching Data from SMHI ===
def fetch_data_from_smhi(station_id):
    parameters = {
        'Vindriktning': 3,
        'Vindhastighet': 4,
        'Lufttryck': 9,
        'Relativ luftfuktighet': 6,
        'Lufttemperatur': 1,
        'Total molnmängd': 2,
        'Nederbörd': 5
    }
    
    all_data = {}
    print(f"\nFetching data for the latest months for station {station_id}...\n")
    
    for name, param_id in parameters.items():
        url = f"https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{param_id}/station/{station_id}/period/latest-months/data.json"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                values = data.get('value', [])

                if values:
                    df = pd.DataFrame(values)
                    df['timestamp'] = pd.to_datetime(df['date'], unit='ms')
                    df = df[['timestamp', 'value']]
                    df.rename(columns={'value': name}, inplace=True)
                    all_data[name] = df
                    print(f"{name}: Retrieved {len(df)} observations.")
                else:
                    raise ValueError("No 'value' data present.")
            elif response.status_code == 404:
                raise ValueError("HTTP 404 – Not Found")
            else:
                raise ValueError(f"HTTP error {response.status_code}")

        except Exception as e:
            print(f"{name}: Error fetching data: {e}")

            # Handling defaults
            if name == 'Nederbörd':
                print(f"{name}: Filling with 0.0 mm")
                reference_df = next(iter(all_data.values()))  # use the first valid reference df
                fallback_df = pd.DataFrame({
                    'timestamp': reference_df['timestamp'],
                    name: [0.0] * len(reference_df)
                })
                all_data[name] = fallback_df

            elif name == 'Total molnmängd':
                print(f"{name}: Filling with 100%")
                reference_df = next(iter(all_data.values()))
                fallback_df = pd.DataFrame({
                    'timestamp': reference_df['timestamp'],
                    name: [100] * len(reference_df)
                })
                all_data[name] = fallback_df

    # Merge all parameters into df_months
    df_months = None
    for df in all_data.values():
        if df_months is None:
            df_months = df
        else:
            df_months = pd.merge(df_months, df, on='timestamp', how='outer')

    # Sort the DataFrame by timestamp and reset the index
    df_months.sort_values('timestamp', inplace=True)
    df_months.reset_index(drop=True, inplace=True)

    return df_months

# === 2. Preprocessing ===
def preprocess_data(df_months):
    df_months.set_index('timestamp', inplace=True)

    df_months['Vindriktning'] = pd.to_numeric(df_months['Vindriktning'], errors='coerce')
    df_months['Vindhastighet'] = pd.to_numeric(df_months['Vindhastighet'], errors='coerce')
    df_months['Lufttryck'] = pd.to_numeric(df_months['Lufttryck'], errors='coerce')
    df_months['Relativ luftfuktighet'] = pd.to_numeric(df_months['Relativ luftfuktighet'], errors='coerce')
    df_months['Lufttemperatur'] = pd.to_numeric(df_months['Lufttemperatur'], errors='coerce')

    df_months.fillna({
        'Vindriktning': 0,
        'Vindhastighet': 0,
        'Lufttryck': df_months['Lufttryck'].mean(),
        'Relativ luftfuktighet': df_months['Relativ luftfuktighet'].mean(),
        'Lufttemperatur': df_months['Lufttemperatur'].mean(),
    }, inplace=True)

    df_daily = df_months.resample('D').agg({
        'Vindriktning': 'mean',
        'Vindhastighet': 'mean',
        'Lufttryck': 'mean',
        'Relativ luftfuktighet': 'mean',
        'Lufttemperatur': 'mean',
        'Total molnmängd': 'mean',
        'Nederbörd': 'sum'
    })

    df_daily['Rain_Yes_No'] = (df_daily['Nederbörd'] > 0).astype(int)

    df_daily.rename(columns={
        'Lufttryck': 'Lufttryck reducerat havsytans nivå',
        'Relativ luftfuktighet': 'Relativ Luftfuktighet',
        'Nederbörd': 'Nederbördsmängd'
    }, inplace=True)

    return df_daily

# === 3. Prepare Lagged Features ===
def prepare_lagged_features(df_daily):
    lags = [1, 2]
    for lag in lags:
        df_daily[f'Vindriktning_lag{lag}'] = df_daily['Vindriktning'].shift(lag)
        df_daily[f'Vindhastighet_lag{lag}'] = df_daily['Vindhastighet'].shift(lag)
        df_daily[f'Lufttryck_lag{lag}'] = df_daily['Lufttryck reducerat havsytans nivå'].shift(lag)
        df_daily[f'Total_molnmängd_lag{lag}'] = df_daily['Total molnmängd'].shift(lag)
        df_daily[f'Relativ_Luftfuktighet_lag{lag}'] = df_daily['Relativ Luftfuktighet'].shift(lag)
        df_daily[f'Lufttemperatur_lag{lag}'] = df_daily['Lufttemperatur'].shift(lag)

    df_daily.dropna(inplace=True)
    return df_daily

# === 4. Load Scalers and Model ===
def load_scalers_and_model():
    scaler_X = joblib.load("./models/scaler_X.pkl")
    scaler_rain_amt = joblib.load("./models/scaler_rain_amount.pkl")
    scaler_wind = joblib.load("./models/scaler_wind_velocity.pkl")
    model = load_model("./models/best_lstm_model.h5", compile=False)
    return scaler_X, scaler_rain_amt, scaler_wind, model

# === 5. Prepare Features for LSTM Prediction ===
def prepare_features_for_prediction(df_daily, n_steps=5):
    features = [col for col in df_daily.columns if 'lag' in col or col in [
        'Vindriktning', 'Vindhastighet', 'Lufttryck reducerat havsytans nivå',
        'Total molnmängd', 'Relativ Luftfuktighet', 'Lufttemperatur'
    ]]
    df_recent = df_daily.tail(n_steps)
    X_recent = df_recent[features].values.reshape((1, n_steps, len(features)))
    return X_recent

# === 6. Make Prediction ===
def make_prediction(X_recent, model, scaler_rain_amt, scaler_wind):
    y_pred = model.predict(X_recent)
    y_pred_rain = scaler_rain_amt.inverse_transform(y_pred[:, 0].reshape(-1, 1))
    y_pred_wind = scaler_wind.inverse_transform(y_pred[:, 2].reshape(-1, 1))
    
    

    if y_pred_rain[0][0] >= 0.5:
        rain_prediction = "rain"
    else:
        rain_prediction = "no rain"

    wind_velocity = f"{y_pred_wind[0][0]:.1f}"
    
    return rain_prediction, wind_velocity #y_pred_wind

# === 7. Main Entry Point ===
if __name__ == "__main__":
    station_id = 66110  # Södra Öland
    
    # Fetch data
    df_months = fetch_data_from_smhi(station_id)
    
    # Preprocess data
    df_daily = preprocess_data(df_months)
    
    # Prepare lagged features
    df_daily = prepare_lagged_features(df_daily)
    
    # Load scalers and model
    scaler_X, scaler_rain_amt, scaler_wind, model = load_scalers_and_model()
    
    # Prepare features for prediction
    X_recent = prepare_features_for_prediction(df_daily, n_steps=5)
    
    # Make prediction
    rain_prediction, wind_velocity = make_prediction(X_recent, model, scaler_rain_amt, scaler_wind)
    
    # Output predictions
    print(f"Predicted Rain_Yes_No_tomorrow: {rain_prediction}")
    print(f"Predicted Wind_Velocity_tomorrow: {wind_velocity} m/s")

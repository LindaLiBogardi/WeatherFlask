import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model
import joblib
import os

# === Load and Preprocess Data ===
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['Datum_Tid'] = pd.to_datetime(df['Datum_Tid'])
    df.set_index('Datum_Tid', inplace=True)
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    df['Hour'] = df.index.hour

    selected_features = [
        'Vindriktning', 'Vindhastighet', 'Lufttryck reducerat havsytans nivå',
        'Total molnmängd', 'Relativ Luftfuktighet', 'Lufttemperatur', 'Nederbördsmängd'
    ]

    df_daily = df[selected_features].resample('D').mean()
    df_daily['YearMonth'] = df_daily.index.to_period('M')

    df_daily['Rain_Yes_No'] = (df_daily['Nederbördsmängd'] > 0.0).astype(np.float64)
    df_daily['Rain_Amount'] = df_daily['Nederbördsmängd']
    df_daily['Wind_Velocity'] = df_daily['Vindhastighet']

    # Keep only April - October
    df_daily = df_daily[df_daily.index.month.isin([4,5,6,7,8,9,10])]

    # Lags
    lags = [1, 2]
    for lag in lags:
        df_daily[f'Vindriktning_lag{lag}'] = df_daily['Vindriktning'].shift(lag)
        df_daily[f'Vindhastighet_lag{lag}'] = df_daily['Vindhastighet'].shift(lag)
        df_daily[f'Lufttryck_lag{lag}'] = df_daily['Lufttryck reducerat havsytans nivå'].shift(lag)
        df_daily[f'Total_molnmängd_lag{lag}'] = df_daily['Total molnmängd'].shift(lag)
        df_daily[f'Relativ_Luftfuktighet_lag{lag}'] = df_daily['Relativ Luftfuktighet'].shift(lag)
        df_daily[f'Lufttemperatur_lag{lag}'] = df_daily['Lufttemperatur'].shift(lag)

    df_daily['Rain_Yes_No_tomorrow'] = df_daily['Rain_Yes_No'].shift(-1)
    df_daily['Rain_Amount_tomorrow'] = df_daily['Rain_Amount'].shift(-1)
    df_daily['Wind_Velocity_tomorrow'] = df_daily['Wind_Velocity'].shift(-1)

    df_daily.dropna(inplace=True)
    return df_daily

# === Feature Engineering ===
def prepare_features(df):
    features = [col for col in df.columns if 'lag' in col or col in [
        'Vindriktning', 'Vindhastighet', 'Lufttryck reducerat havsytans nivå',
        'Total molnmängd', 'Relativ Luftfuktighet', 'Lufttemperatur'
    ]]
    targets = ['Rain_Yes_No_tomorrow', 'Rain_Amount_tomorrow', 'Wind_Velocity_tomorrow']

    # Scale features
    scaler_X = MinMaxScaler()
    df[features] = scaler_X.fit_transform(df[features])

    # Scale targets
    scaler_rain_amt = MinMaxScaler()
    df['Rain_Amount_tomorrow'] = scaler_rain_amt.fit_transform(df[['Rain_Amount_tomorrow']])

    scaler_wind = MinMaxScaler()
    df['Wind_Velocity_tomorrow'] = scaler_wind.fit_transform(df[['Wind_Velocity_tomorrow']])

    # 💾 Save scalers
    os.makedirs("./models", exist_ok=True)
    joblib.dump(scaler_X, "./models/scaler_X.pkl")
    joblib.dump(scaler_rain_amt, "./models/scaler_rain_amount.pkl")
    joblib.dump(scaler_wind, "./models/scaler_wind_velocity.pkl")
    print("✅ Scalers saved to ./models/")

    return df, features, targets, scaler_rain_amt, scaler_wind

# === Sequence Creation ===
def build_sequences(df, features, targets, n_steps=5):
    X, y = [], []
    for i in range(n_steps, len(df)):
        X.append(df[features].iloc[i-n_steps:i].values)
        y.append(df[targets].iloc[i].values)
    return np.array(X), np.array(y)

# === Binary Classifier with Class Weights ===
def train_binary_classifier(X, y_binary):
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, shuffle=False)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, class_weight=class_weights_dict)
    return model, X_test, y_test

# === Hyperparameter Tuning Loop ===
def hyperparameter_tuning(X_train, y_train, X_test, y_test):
    param_grid = {
        'units': [50],
        'dropout_rate': [0.2],
        'optimizer': ['adam'],
        'batch_size': [32],
        'epochs': [10]
    }

    best_model = None
    best_result = None
    best_recall = -1

    for units in param_grid['units']:
        for dropout in param_grid['dropout_rate']:
            for optimizer in param_grid['optimizer']:
                for batch_size in param_grid['batch_size']:
                    for epochs in param_grid['epochs']:
                        print(f"🔧 Training model: units={units}, dropout={dropout}, optimizer={optimizer}, batch_size={batch_size}, epochs={epochs}")

                        model = Sequential()
                        model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                        model.add(Dropout(dropout))
                        model.add(LSTM(units))
                        model.add(Dropout(dropout))
                        model.add(Dense(3, activation='linear'))

                        model.compile(optimizer=optimizer, loss='mse')
                        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)

                        y_pred = model.predict(X_test)

                        y_true_bin = y_test[:, 0]
                        y_pred_bin = (y_pred[:, 0] > 0.3).astype(int)

                        recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
                        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

                        print(f"Recall: {recall:.4f} | F1: {f1:.4f}")

                        if recall > best_recall:
                            best_recall = recall
                            best_model = model
                            best_result = {
                                'units': units, 'dropout': dropout, 'optimizer': optimizer,
                                'batch_size': batch_size, 'epochs': epochs, 'recall': recall, 'f1': f1
                            }

    if best_model:
        print("✅ Best model based on Recall:")
        for k, v in best_result.items():
            print(f"{k}: {v}")
        os.makedirs("models", exist_ok=True)
        best_model.save("models/best_lstm_model.keras")  

    else:
        print("⚠️ No model with positive recall found.")
    return best_model

# === Main Entry Point ===
if __name__ == "__main__":
    filepath = "FilledCleanDataMoln.csv"
    df = load_and_preprocess_data(filepath)
    df, features, targets, scaler_rain_amt, scaler_wind = prepare_features(df)

    n_steps = 5
    X, y = build_sequences(df, features, targets, n_steps=n_steps)

    # Optionally, run binary classification model with class weights
    y_binary = df['Rain_Yes_No_tomorrow'].values[n_steps:]
    model_binary, X_bin_test, y_bin_test = train_binary_classifier(X, y_binary)

    # Hyperparameter tuning and save best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    best_model = hyperparameter_tuning(X_train, y_train, X_test, y_test)

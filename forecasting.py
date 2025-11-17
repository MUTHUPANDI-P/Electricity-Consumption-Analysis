import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from eda import load_data

# Create features for forecasting
def create_features(df):
    df['lag_1'] = df['Global_active_power'].shift(1)
    df['lag_2'] = df['Global_active_power'].shift(2)
    df.dropna(inplace=True)
    return df


def train_test_split(df):
    train = df.iloc[:-500]
    test = df.iloc[-500:]
    return train, test


def train_model(train):
    model = RandomForestRegressor(n_estimators=200)
    X_train = train[['lag_1', 'lag_2']]
    y_train = train['Global_active_power']
    model.fit(X_train, y_train)
    return model


def evaluate(model, test):
    X_test = test[['lag_1', 'lag_2']]
    y_test = test['Global_active_power']

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("MAE:", mae)
    print("RMSE:", rmse)

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:500], label="Actual")
    plt.plot(predictions[:500], label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted Global Active Power")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    df = create_features(df)
    train, test = train_test_split(df)
    model = train_model(train)
    evaluate(model, test)

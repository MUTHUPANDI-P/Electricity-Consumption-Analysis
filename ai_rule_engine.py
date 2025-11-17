import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from eda import load_data

def detect_anomalies(daily_df):
    iso = IsolationForest(contamination=0.02, random_state=42)
    daily_df["Anomaly"] = iso.fit_predict(daily_df[['DailyConsumption']])

    plt.figure(figsize=(10, 5))
    plt.scatter(
        daily_df.index,
        daily_df['DailyConsumption'],
        c=daily_df["Anomaly"],
        cmap='coolwarm'
    )
    plt.title("Anomaly Detection in Daily Consumption")
    plt.xlabel("Date")
    plt.ylabel("Consumption")
    plt.show()

    return daily_df


def consumption_category(value):
    if value < 1.5:
        return "Low Usage", "Try switching to energy-efficient appliances!"
    elif value < 3.0:
        return "Medium Usage", "Turn off unused devices to save energy."
    else:
        return "High Usage", "Consider reducing AC/heater usage to save power."


if __name__ == "__main__":
    df = load_data()
    daily_df = df['Global_active_power'].resample('D').sum().to_frame(name='DailyConsumption')

    detect_anomalies(daily_df)

    latest_value = df['Global_active_power'].iloc[-1]
    category, advice = consumption_category(latest_value)

    print("Latest Consumption:", latest_value)
    print("Category:", category)
    print("AI Suggestion:", advice)

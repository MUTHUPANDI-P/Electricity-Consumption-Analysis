import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    path = "../data/household_power_consumption.txt"

    df = pd.read_csv(
        path,
        sep=";",
        low_memory=False,
        na_values="?"
    )

    print("Dataset Loaded Successfully!")

    # Combine date & time
    df['Datetime'] = pd.to_datetime(
        df['Date'] + " " + df['Time'],
        errors='coerce'
    )
    df.sort_values('Datetime', inplace=True)
    df.set_index('Datetime', inplace=True)

    # Convert numeric columns
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity',
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df


def plot_global_power(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df['Global_active_power'], linewidth=0.5)
    plt.title("Global Active Power Over Time")
    plt.xlabel("Time")
    plt.ylabel("Kilowatts")
    plt.show()


def plot_daily_average(df):
    daily = df['Global_active_power'].resample('D').mean()

    plt.figure(figsize=(12, 5))
    plt.plot(daily)
    plt.title("Daily Global Active Power")
    plt.xlabel("Date")
    plt.ylabel("Kilowatts")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    print(df.isnull().sum())
    plot_global_power(df)
    plot_daily_average(df)

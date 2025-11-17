import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from eda import load_data

def daily_consumption(df):
    daily_df = df['Global_active_power'].resample('D').sum().dropna()
    daily_df = daily_df.to_frame(name="DailyConsumption")
    return daily_df


def cluster_daily_usage(daily_df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    daily_df["Cluster"] = kmeans.fit_predict(daily_df[['DailyConsumption']])

    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        x=daily_df.index,
        y="DailyConsumption",
        hue="Cluster",
        data=daily_df
    )
    plt.title("Daily Consumption Clusters")
    plt.xlabel("Date")
    plt.ylabel("Consumption")
    plt.show()

    return daily_df


if __name__ == "__main__":
    df = load_data()
    daily_df = daily_consumption(df)
    cluster_daily_usage(daily_df)

from eda import load_data, plot_global_power, plot_daily_average
from forecasting import create_features, train_test_split, train_model, evaluate
from clustering import daily_consumption, cluster_daily_usage
from ai_rule_engine import detect_anomalies, consumption_category


def main():
    print("ELECTRICITY CONSUMPTION ANALYSIS PIPELINE")

    df = load_data()
    plot_global_power(df)
    plot_daily_average(df)

    df_feat = create_features(df)
    train, test = train_test_split(df_feat)
    model = train_model(train)
    evaluate(model, test)

    daily_df = daily_consumption(df)
    cluster_daily_usage(daily_df)

    detect_anomalies(daily_df)

    latest_value = df['Global_active_power'].iloc[-1]
    category, suggestion = consumption_category(latest_value)

    print("Latest Global Active Power:", latest_value)
    print("Usage Category:", category)
    print("AI Suggestion:", suggestion)

    print("Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()

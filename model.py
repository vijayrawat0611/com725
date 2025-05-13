#!/usr/bin/env python
# coding: utf-8

# In[77]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import subprocess



from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
st.set_page_config(page_title="CRYPTOCURRENCY", layout="wide")
st.title("📊 CRYPTOCURRENCY  Data Analysis Dashboard")
st.sidebar.header("Navigation")


st.title("🏠 Home")

if st.button("Go to Home Page"):
    subprocess.run(["streamlit", "run", "crypto.py"])


option1 = st.selectbox("Select COIN", ["BITCOIN", "ETHEREUM", "DOGECOIN","LITECOIN"])
option2 = st.selectbox("Select MODEL", ["RandomForest","SVM", "Linear Regression","Decision Tree"])

if option1 == "BITCOIN" and option2 == "RandomForest":
    
    df = pd.read_csv("bitcoin.csv")
    df['Date'] = pd.to_datetime(df['Start'], format="%d/%m/%Y")
    df.sort_values('Date', inplace=True)
    
    st.header("RandomForest Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))
    

elif option1 == "ETHEREUM" and option2 == "RandomForest":
    df = pd.read_csv("ethereum.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    st.header("RandomForest Model Forecast")

    
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))




elif option1 == "DOGECOIN" and option2 == "RandomForest":
    df = pd.read_csv("dogecoin.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("RandomForest Model Forecast")

    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))



elif option1 == "LITECOIN" and option2 == "RandomForest":
    df = pd.read_csv("litecoin.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("RandomForest Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))



















elif option1 == "BITCOIN" and option2 == "SVM":
    
    df = pd.read_csv("bitcoin.csv")
    df['Date'] = pd.to_datetime(df['Start'], format="%d/%m/%Y")
    df.sort_values('Date', inplace=True)
    
    st.header("SVM Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = SVR(kernel='rbf', C=1000, epsilon=0.1)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))
    

elif option1 == "ETHEREUM" and option2 == "SVM":
    df = pd.read_csv("ethereum.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("SVM Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = SVR(kernel='rbf', C=1000, epsilon=0.1)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))

elif option1 == "DOGECOIN" and option2 == "SVM":
    df = pd.read_csv("dogecoin.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("SVM Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = SVR(kernel='rbf', C=1000, epsilon=0.1)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))

elif option1 == "LITECOIN" and option2 == "SVM":
    df = pd.read_csv("litecoin.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("SVM Model Forecast")

    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = SVR(kernel='rbf', C=1000, epsilon=0.1)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))











# linear regression

elif option1 == "BITCOIN" and option2 == "Linear Regression":
    
    df = pd.read_csv("bitcoin.csv")
    df['Date'] = pd.to_datetime(df['Start'], format="%d/%m/%Y")
    df.sort_values('Date', inplace=True)
    
    st.header("Regression Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = SVR(kernel='rbf', C=1000, epsilon=0.1)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))
    

elif option1 == "ETHEREUM" and option2 == "Linear Regression":
    df = pd.read_csv("ethereum.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("Regression Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))


elif option1 == "DOGECOIN" and option2 == "Linear Regression":
    df = pd.read_csv("dogecoin.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("Regression Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))











elif option1 == "LITECOIN" and option2 == "Linear Regression":
    df = pd.read_csv("litecoin.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("Regression Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))





elif option1 == "BITCOIN" and option2 == "Decision Tree":
    
    df = pd.read_csv("bitcoin.csv")
    df['Date'] = pd.to_datetime(df['Start'], format="%d/%m/%Y")
    df.sort_values('Date', inplace=True)
    
    st.header("Decision Tree Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))
    

elif option1 == "ETHEREUM" and option2 == "Decision Tree":
    df = pd.read_csv("ethereum.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("Decision Tree Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))




elif option1 == "DOGECOIN" and option2 == "Decision Tree":
    df = pd.read_csv("dogecoin.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("Decision Tree Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))




elif option1 == "LITECOIN" and option2 == "Decision Tree":
    df = pd.read_csv("litecoin.csv")
    df["Date"] = pd.to_datetime(df["Start"], format="%d/%m/%Y")

    df.sort_values('Date', inplace=True)
    
    st.header("Decision Tree Model Forecast")
    st.write("Latest data preview:")
    st.dataframe(df.tail())

    # Sidebar controls
    n_lags = st.sidebar.slider("Number of Lag Days", 3, 30, 7, key="slider_lags")
    n_forecast = st.sidebar.radio("Forecast Horizon (days)", [7, 15], key="radio_forecast")

    # Feature engineering
    def create_lag_features(data, n_lags):
        for lag in range(1, n_lags + 1):
            data[f"lag_{lag}"] = data["Close"].shift(lag)
        data.dropna(inplace=True)
        return data

    df = create_lag_features(df, n_lags)

    feature_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    X = df[feature_cols]
    y = df['Close']  # Predict actual prices, not z-scores

    # Train model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Forecast loop
    last_row = X.iloc[-1].values
    last_row = pd.DataFrame([last_row], columns=feature_cols)  # Convert to DataFrame with correct column names
    future_predictions = []

    for _ in range(n_forecast):
        pred = model.predict(last_row)[0]
        future_predictions.append(pred)
        last_row = np.roll(last_row.values, -1)
        last_row[0, -1] = pred
        last_row = pd.DataFrame([last_row[0]], columns=feature_cols)

    # Forecast dates
    last_date = df["Date"].iloc[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_forecast)

    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_predictions})

    # Combine actual and forecast data
    full_df = pd.concat([
        df[["Date", "Close"]].rename(columns={"Close": "Price"}),
        forecast_df.rename(columns={"Forecast": "Price"})
    ])

    # Plotting
    st.subheader(f"📉 Forecast for Next {n_forecast} Days")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(full_df["Date"], full_df["Price"], label="BTC Price", color="blue")
    ax.axvline(x=forecast_df["Date"].iloc[0], color="gray", linestyle="--", label="Forecast Start")
    ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Forecast", color="orange", linestyle="--", marker="o")
    ax.set_title("BTC Price Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.grid(True)
    ax.legend()

    # Display the plot with Streamlit
    st.pyplot(fig)  # This will properly render the plot

    # Display forecasted data
    st.write("Forecasted Prices:")
    st.dataframe(forecast_df.set_index("Date"))
    
else:
    st.info("Waiting for correct input...")


# In[ ]:





# In[ ]:





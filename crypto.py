#!/usr/bin/env python
# coding: utf-8

# In[48]:


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import subprocess

st.set_page_config(page_title="CRYPTOCURRENCY", layout="wide")
st.title("📊 CRYPTOCURRENCY  Data Analysis Dashboard")
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["ABOUT US","EDA", "PCA","MODEL"])



if section == "ABOUT US":
    st.write("Solent Intelligence (SOLiGence) is a leading financial multinational organisation that deals with stock and shares, cryptocurrencies and investments. The organisation operates an online investment platform that accommodates millions of subscribers (clients) with over 150 billion pounds worth of investments. SOLiGence is a secure organisation that trades on multiple stock exchange platforms such as FTSE 100, equities, cryptocurrencies and other commodities. Due to the large customer base and competition from newly sprung up fintech organisations, the company intend to implement an Intelligent Coin Trading (ICT) platform with an emphasis on crypto coin predictions. The ICT platform will perform intelligent trading by allowing purchases of CRYPTOCURRENCIES at lower prices and sells at higher prices to realise a substantive profit. The platform will perhaps anticipate prices of coins on a daily, weekly, monthly and the maximum of a quarterly basis. (i.e., the system will identify crypto coins with the potential of buying low and selling high at the specified interval ")


if section == "EDA":
    coin = st.selectbox("SELECT COIN", ["BITCOIN", "ETHEREUM", "DOGECOIN", "LITECOIN"])
    if coin == "BITCOIN":
        df = pd.read_csv("bitcoin.csv")
        st.header("Exploratory Data Analysis/ BITCOIN")
        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Summary Statistics")
        st.write(df.describe())
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected_col = st.selectbox("Select column for distribution plot", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)
    elif coin == "ETHEREUM":
        df = pd.read_csv("ethereum.csv")
        st.header("Exploratory Data Analysis/ETHEREUM")
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        st.subheader("Data Types")
        st.write(df.dtypes)
        st.subheader("Summary Statistics")
        st.write(df.describe())
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected_col = st.selectbox("Select column for distribution plot", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

    elif coin == "DOGECOIN":
        df = pd.read_csv("dogecoin.csv")
        st.header("Exploratory Data Analysis/DOGECOIN")
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        st.subheader("Data Types")
        st.write(df.dtypes)
        st.subheader("Summary Statistics")
        st.write(df.describe())
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected_col = st.selectbox("Select column for distribution plot", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

    elif coin == "LITECOIN":
        df = pd.read_csv("litecoin.csv")
        st.header("Exploratory Data Analysis/LITECOIN")
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Summary Statistics")
        st.write(df.describe())
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=np.number).columns
        selected_col = st.selectbox("Select column for distribution plot", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

elif section == "PCA":
    currency = st.selectbox("SELECT CRYPTO", ["BITCOIN", "ETHEREUM", "DOGECOIN", "LITECOIN"])
    if currency == "BITCOIN":
        df = pd.read_csv("bitcoin.csv")
        st.header("Principal Component Analysis/BITCOIN")
        numeric_data = df.select_dtypes(include=np.number).dropna()
        n_components = st.slider("Number of components", 2, min(10, numeric_data.shape[1]), 2)

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_data)
        pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])

        st.subheader("Explained Variance Ratio")
        st.bar_chart(pca.explained_variance_ratio_)

        st.subheader("PCA Scatter Plot (PC1 vs PC2)")
        fig, ax = plt.subplots()
        sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], ax=ax)
        st.pyplot(fig)
    elif currency == "ETHEREUM":
        df = pd.read_csv("ethereum.csv")
        st.header("Principal Component Analysis/ETHEREUM")
        numeric_data = df.select_dtypes(include=np.number).dropna()
        n_components = st.slider("Number of components", 2, min(10, numeric_data.shape[1]), 2)

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_data)
        pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])

        st.subheader("Explained Variance Ratio")
        st.bar_chart(pca.explained_variance_ratio_)

        st.subheader("PCA Scatter Plot (PC1 vs PC2)")
        fig, ax = plt.subplots()
        sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], ax=ax)
        st.pyplot(fig)
    elif currency == "DOGECOIN":
        df = pd.read_csv("dogecoin.csv")
        st.header("Principal Component Analysis/DOGECOIN")
        numeric_data = df.select_dtypes(include=np.number).dropna()
        n_components = st.slider("Number of components", 2, min(10, numeric_data.shape[1]), 2)

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_data)
        pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])

        st.subheader("Explained Variance Ratio")
        st.bar_chart(pca.explained_variance_ratio_)

        st.subheader("PCA Scatter Plot (PC1 vs PC2)")
        fig, ax = plt.subplots()
        sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], ax=ax)
        st.pyplot(fig)

    elif currency == "LITECOIN":
        df = pd.read_csv("litecoin.csv")
        st.header("Principal Component Analysis/LITECOIN")
        numeric_data = df.select_dtypes(include=np.number).dropna()
        n_components = st.slider("Number of components", 2, min(10, numeric_data.shape[1]), 2)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(numeric_data)
        pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components)])
        st.subheader("Explained Variance Ratio")
        st.bar_chart(pca.explained_variance_ratio_)
        st.subheader("PCA Scatter Plot (PC1 vs PC2)")
        fig, ax = plt.subplots()
        sns.scatterplot(x=pca_df["PC1"], y=pca_df["PC2"], ax=ax)
        st.pyplot(fig)

        
         

elif section == "MODEL":
    subprocess.run(["streamlit", "run", "model.py"])
        
        
            
        
   
            

    




# In[ ]:






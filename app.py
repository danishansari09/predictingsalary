import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()

st.title('Salary Predictor')
sal_data = pd.read_csv('data//Salary_Data.csv')
X = sal_data["YearsExperience"].values
y = sal_data["Salary"].values

lr = LinearRegression()
lr.fit(X.reshape(-1,1), y)

radio = st.sidebar.radio("Navigation",["Home",'Predictor'])

if radio == "Home":
    st.image("data//sal.jpg", width=600)
    chk_tbl = st.checkbox("Show Salary Data")
    if chk_tbl:
        st.table(sal_data)
    graph = st.selectbox("Select kind of graph: ", ["Non-Interactive","Interactive"])

    yrs = st.slider("Number of Years: ", min_value=0, max_value=20, value=5)
    sal_data = sal_data.loc[sal_data["YearsExperience"] >= yrs]
    fig_, ax = plt.subplots()
    if graph == "Non-Interactive":
        sns.scatterplot(sal_data['YearsExperience'], sal_data['Salary'], ax=ax) 
        st.pyplot(fig_)
    else:
        plt.figure(figsize=(10,8))
        # layout = go.Layout(
        #     xaxis= dict(range=[0,12]),
        #     yaxis= dict(range=(0,200000))
        # )
        fig = go.Figure(data=go.Scatter(x=sal_data['YearsExperience'], y=sal_data['Salary'], mode='markers'))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.header("Know your Salary")
    num_yrs = st.number_input("Enter years of experience: ", min_value=0.00, max_value=20.00, step=0.25)
    num_yrs = np.array(num_yrs).reshape(-1,1)
    pred = lr.predict(num_yrs)[0]
    if st.button("Predict"):
        st.success(f"Your predicted annual salary is ${round(pred, 2)}")
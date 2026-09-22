import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

data = pd.read_csv("IceCreamData.csv")

x = data.iloc[: ,[0]]
y = data.iloc[: ,1]

x_train , x_test , y_train , y_test = train_test_split(x,y,train_size=0.8)

model = LinearRegression()
model.fit(x_train , y_train)

train_score = model.score(x_train ,y_train)
test_score = model.score(x_test ,y_test)


# app code :
st.title("Ice Cream Revenue Prediction")

st.metric(label="Model Accuracy ", value=f"{test_score* 100:.2f}%")
temperature = st.slider("Select Temperature (°C)", min_value=0.0, max_value=55.0, value=25.0)

input_data = pd.DataFrame({x.columns[0]: [temperature]})
prediction = model.predict(input_data)[0]

st.success(f"Predicted Revenue: ${prediction:.2f}")

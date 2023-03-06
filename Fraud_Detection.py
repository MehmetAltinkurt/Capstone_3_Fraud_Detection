import streamlit as st

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
from PIL import Image
import pickle

from xgboost import XGBClassifier

st.set_page_config(page_title="Group 8 Fraud Detection Project",layout="wide" )


st.markdown(
    """
    <style>

    {
       background: #ffff99; 
       background: -webkit-linear-gradient(to right, #ff0099, #493240); 
       background: linear-gradient(to right, #ff0099, #493240); 
    }

    </style>
    """,
    unsafe_allow_html=True,
)



hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
              background: #b8d2d9; 
              background: -webkit-linear-gradient(to down, #ff0099, #493240); 
              background: linear-gradient(to down, #ff0099, #493240); 
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
#title text
st.markdown("<h1 style='text-align: center; color: purple;'>Group 8</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: purple;'>Fraud Detection</h1>", unsafe_allow_html=True)


st.title(" ")

Amount = st.sidebar.slider("Amount", min_value=0.00, max_value=25691.16, value=22.00, step=0.01)
V4=st.sidebar.slider("V4", min_value=-5.68, max_value=16.88, value=-0.02, step=0.01)
st.sidebar.markdown("""---""")
V10=st.sidebar.slider("V10", min_value=-24.59, max_value=23.75, value=-0.09, step=0.01)
st.sidebar.markdown("""---""")
V13=st.sidebar.slider("V13", min_value=-5.79, max_value=7.13, value=-0.01, step=0.01)
st.sidebar.markdown("""---""")
V14=st.sidebar.slider("V14", min_value=-19.21, max_value=10.53, value=0.05, step=0.01)
st.sidebar.markdown("""---""")
V17=st.sidebar.slider("V17", min_value=-25.16, max_value=9.25, value=-0.07, step=0.01)
st.sidebar.markdown("""---""")


st.markdown("<h3 style='text-align: left; color: black;'>Model Karşılaştırma Grafiği</h3>", unsafe_allow_html=True)
st.image("model_karsilastir.png", caption="Model Karşilaştirma Grafiği", width=1200, use_column_width=None, clamp=False, channels="RGB", output_format="auto")





my_dict = {
    "Time":0,
    "V1":0,
    "V2":0,
    "V3":0,
    "V4":V4,
    "V5":0,
    "V6":0,
    "V7":0,
    "V8":0,
    "V9":0,
    "V10":V10,
    "V11":0,
    "V12":0,
    "V13":V13,
    "V14":V14,
    "V15":0,
    "V16":0,
    "V17":V17,
    "V18":0,
    "V19":0,
    "V20":0,
    "V21":0,
    "V22":0,
    "V23":0,
    "V24":0,
    "V25":0,
    "V26":0,
    "V27":0,
    "V28":0,
    "Amount": Amount
}

scaler = pickle.load(open('RobustScaler', 'rb'))
model = pickle.load(open('XGB', 'rb'))


st.markdown("<h3 style='text-align: left; color: black;'>Prediction</h3>", unsafe_allow_html=True)
df=pd.DataFrame.from_dict([my_dict])
st.table(df[["V4","V10","V13","V14","V17","Amount"]])


if st.button("Predict"):
    X=df.copy()
    X_scaled = scaler.transform(X)
    X_scaled=X_scaled[0,[4,10,13,14,17,29]]
    #st.success("Fraud status: "+str(X_scaled))
    y_pred = model.predict(X_scaled.reshape(1,6))
    if y_pred[0]==0:
       st.success("Not Fraud")
    else:
       st.error("Fraud")
    





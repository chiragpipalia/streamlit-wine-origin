import streamlit as st
import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier

st.title('Wine Origin')

#Loading up the Regression model we created
model = XGBClassifier()
model.load_model('wine-origin-streamlit/xgb_model.json')
# model.load_model('xgb_model.json')

def load_data():
	df = pd.read_csv('wine-origin-streamlit/wine-dataset/wine.data')
	columns = ['Target','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols'
	      ,'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'] 
	df.columns = columns
	#df['Target'] = df['Target'].map({1:0, 2:1, 3:2})
	df = df[['Alcohol','Malic acid','Ash','Alcalinity of ash']].copy()
	return df

data = load_data()

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

## Get inputs
od_filter = st.slider('OD280 of Diluted Wine', min_value=1.3, max_value=4.0, step=0.1)
color_intensity_filter = st.slider('Color Intensity', min_value=1.3, max_value=13.0, step=0.1)
flavanoids_filter = st.slider('Flavanoids', min_value=0.3, max_value=5.1, step=0.1)
proline_filter = st.slider('Proline', min_value=278, max_value=1680, step=50)


# Define the prediction function
def get_wine_origin(alcohol, malic_acid, ash, alcalinity):
	st.write('Gathering ingredients... :sunglasses:')
	st.text('Brewing...')

	time.sleep(1)
	data = pd.DataFrame([[alcohol, malic_acid, ash, alcalinity]], columns=['OD280/OD315 of diluted wines', 'Color intensity', 'Flavanoids', 'Proline'])
	prediction = model.predict(data)
	st.text('Brewing...')
	st.write('Prediction: Origin', prediction[0])

get_wine_origin(od_filter, color_intensity_filter, flavanoids_filter, proline_filter)

st.write('Dataset Ref: https://archive.ics.uci.edu/dataset/109/wine')

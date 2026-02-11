
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('linear_regression_model.joblib')

df_original = pd.read_csv('CarPrice.csv')


company_mapping = {
    'vw': 'volkswagen',
    'vokswagen': 'volkswagen',
    'porche': 'porsche',
    'maxda': 'mazda',
    'nissan': 'nissan',
    'toyota': 'toyota'
}
door_mapping = {'two': 2, 'four': 4}
cylinder_mapping = {
    'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'eight': 8, 'twelve': 12
}

def get_model_feature_columns(df_initial):
    df_temp = df_initial.copy()
    df_temp['CarCompany'] = df_temp['CarName'].apply(lambda x: x.split(' ')[0].lower())
    df_temp['CarCompany'] = df_temp['CarCompany'].replace(company_mapping)
    df_temp['doornumber'] = df_temp['doornumber'].replace(door_mapping)
    df_temp['cylindernumber'] = df_temp['cylindernumber'].replace(cylinder_mapping)
    df_temp = df_temp.drop(['CarName', 'car_ID'], axis=1)
    categorical_cols = df_temp.select_dtypes(include=['object']).columns
    df_temp = pd.get_dummies(df_temp, columns=categorical_cols, drop_first=True)
    return df_temp.drop('price', axis=1).columns

model_feature_columns = get_model_feature_columns(df_original)

st.title('Car Price Predictor')
st.markdown("""

### ABOUT

This application is a machine learning–powered web tool designed
to estimate car prices based on user-provided features. By analyzing historical vehicle data
and important attributes such as engine type, mileage, year of manufacture, and condition,
the model delivers data-driven price predictions to assist buyers and sellers in evaluating
vehicles more effectively.

""")
st.write('Enter the car features to get a price prediction.')


def user_input_features():
    symboling = st.sidebar.slider('Symboling (Insurance Risk Rating)', -3, 3, 0)
    fueltype = st.sidebar.selectbox('Fuel Type', df_original['fueltype'].unique())
    aspiration = st.sidebar.selectbox('Aspiration', df_original['aspiration'].unique())
    doornumber = st.sidebar.selectbox('Number of Doors', ['two', 'four'])
    carbody = st.sidebar.selectbox('Car Body', df_original['carbody'].unique())
    drivewheel = st.sidebar.selectbox('Drive Wheel', df_original['drivewheel'].unique())
    enginelocation = st.sidebar.selectbox('Engine Location', df_original['enginelocation'].unique())
    wheelbase = st.sidebar.slider('Wheel Base (inches)', 86.6, 120.9, 95.0)
    carlength = st.sidebar.slider('Car Length (inches)', 141.1, 208.1, 170.0)
    carwidth = st.sidebar.slider('Car Width (inches)', 60.3, 72.3, 65.0)
    carheight = st.sidebar.slider('Car Height (inches)', 47.8, 59.8, 53.0)
    curbweight = st.sidebar.slider('Curb Weight (pounds)', 1488, 4066, 2500)
    enginetype = st.sidebar.selectbox('Engine Type', df_original['enginetype'].unique())
    cylindernumber = st.sidebar.selectbox('Number of Cylinders', ['two', 'three', 'four', 'five', 'six', 'eight', 'twelve'])
    enginesize = st.sidebar.slider('Engine Size (cubic inches)', 61, 326, 120)
    fuelsystem = st.sidebar.selectbox('Fuel System', df_original['fuelsystem'].unique())
    boreratio = st.sidebar.slider('Bore Ratio', 2.54, 3.94, 3.3)
    stroke = st.sidebar.slider('Stroke', 2.07, 4.17, 3.2)
    compressionratio = st.sidebar.slider('Compression Ratio', 7.0, 23.0, 9.0)
    horsepower = st.sidebar.slider('Horsepower', 48, 288, 100)
    peakrpm = st.sidebar.slider('Peak RPM', 4150, 6600, 5000)
    citympg = st.sidebar.slider('City MPG', 13, 49, 25)
    highwaympg = st.sidebar.slider('Highway MPG', 16, 54, 30)
    carcompany = st.sidebar.selectbox('Car Company', sorted(df_original['CarName'].apply(lambda x: x.split(' ')[0].lower()).replace(company_mapping).unique()))

    data = {
        'symboling': symboling,
        'fueltype': fueltype,
        'aspiration': aspiration,
        'doornumber': doornumber,
        'carbody': carbody,
        'drivewheel': drivewheel,
        'enginelocation': enginelocation,
        'wheelbase': wheelbase,
        'carlength': carlength,
        'carwidth': carwidth,
        'carheight': carheight,
        'curbweight': curbweight,
        'enginetype': enginetype,
        'cylindernumber': cylindernumber,
        'enginesize': enginesize,
        'fuelsystem': fuelsystem,
        'boreratio': boreratio,
        'stroke': stroke,
        'compressionratio': compressionratio,
        'horsepower': horsepower,
        'peakrpm': peakrpm,
        'citympg': citympg,
        'highwaympg': highwaympg,
        'CarCompany': carcompany
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

def preprocess_user_input(input_df_raw, all_feature_columns):

    input_df_raw['CarCompany'] = input_df_raw['CarCompany'].replace(company_mapping)

    input_df_raw['doornumber'] = input_df_raw['doornumber'].replace(door_mapping)
    input_df_raw['cylindernumber'] = input_df_raw['cylindernumber'].replace(cylinder_mapping)


    categorical_cols_to_encode = [
        'fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
        'enginetype', 'fuelsystem', 'CarCompany'
    ]

    # Apply one-hot encoding
    processed_input = pd.get_dummies(input_df_raw, columns=categorical_cols_to_encode, drop_first=True)


    for col in all_feature_columns:
        if col not in processed_input.columns:
            processed_input[col] = 0
    processed_input = processed_input[all_feature_columns]

    return processed_input.astype(float) # Ensure all features are numeric

final_input = preprocess_user_input(input_df.copy(), model_feature_columns)

st.subheader('User Input Features')
st.write(input_df)

if st.button('Predict Price'):
    prediction = model.predict(final_input)
    st.subheader('Predicted Car Price')
    st.success(f'${prediction[0]:,.2f}')

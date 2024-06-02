import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb

def generateDataFrame(csvFileName):
    try:
        df = pd.read_csv(csvFileName)
        if 'Outcome' not in df.columns:
            st.error("Error: 'Outcome' column not found in the CSV file.")
            return None
        df = df.dropna()
        df = df.apply(pd.to_numeric, errors='ignore')
        return df
    except Exception as e:
        st.error(f"An error occurred while generating the DataFrame: {e}")
        return None

def graphY(dataFrame):
    plt.figure()
    plt.title("Total Of Diabetes")
    sb.countplot(dataFrame["Outcome"], label="Count")
    plt.figtext(.02, -0.07, "1 = Diabetes | 0 = Healthy", color="m", size=10)
    plt.xlabel("Negative | Positive")
    st.pyplot(plt)

def randomForest(dataFrame):
    return RandomForestClassifier(n_estimators=500, random_state=0)

def getAccuracy(rf, df):
    try:
        if 'Outcome' not in df.columns:
            st.error("Error: 'Outcome' column not found in the DataFrame.")
            return None, None, None
        X = df.loc[:, df.columns != 'Outcome']
        y = df['Outcome']
        if X.empty or y.empty:
            st.error("Error: Features or target variable is empty.")
            return None, None, None
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)
        rf.fit(X_train, y_train)
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        return rf, train_score, test_score
    except Exception as e:
        st.error(f"An error occurred while getting accuracy: {e}")
        return None, None, None

def main(fileCSV):
    df = generateDataFrame(fileCSV)
    if df is not None:
        rf = randomForest(df)
        rf, train_score, test_score = getAccuracy(rf, df)
        if rf is not None:
            st.write(rf.predict([[6, 148, 72, 35, 0, 33.6, 0.627, 50]]))
            st.write(rf.predict([[1, 85, 66, 29, 0, 26.6, 0.351, 31]]))
            st.write(test_score)
    return 0

def getResult(fileCSV, data):
    df = generateDataFrame(fileCSV)
    if df is not None:
        rf = randomForest(df)
        rf, train_score, test_score = getAccuracy(rf, df)
        if rf is not None:
            prediction = rf.predict([data])[0]
            return prediction, test_score
    return None, None

st.title("Prediksi Diabetes Gestasional")
st.write("By Ayub")

csvFile = "./diabetes.csv"

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    csvFile = uploaded_file

df = generateDataFrame(csvFile)

if df is not None:
    st.write("Data Preview:")
    st.write(df.head())
    
    graphY(df)

    negative = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
    positive = [6, 148, 72, 35, 0, 33.6, 0.627, 50]

    with st.form(key='Prediksi Diabetes'):
        st.write("Masukan data untuk prediksi:")
        pregnancies = st.number_input('Kehamilan', min_value=0, value=1)
        glucose = st.number_input('Glucose', min_value=0, value=85)
        blood_pressure = st.number_input('Blood Pressure', min_value=0, value=66)
        skin_thickness = st.number_input('Skin Thickness', min_value=0, value=29)
        insulin = st.number_input('Insulin', min_value=0, value=0)
        bmi = st.number_input('BMI', min_value=0.0, value=26.6)
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.351)
        age = st.number_input('Umur', min_value=0, value=31)
        submit_button = st.form_submit_button(label='Predict')
    
    if submit_button:
        data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        classification, accuracy = getResult(csvFile, data)
        if classification is not None and accuracy is not None:
            st.write(f"This machine learning accuracy is {accuracy*100:.2f}%")
            st.write(f"With the data: {data}")
            st.write(f"This machine learning classification of diabetes is {'positive' if classification else 'negative'}")
        else:
            st.error("An error occurred during processing.")


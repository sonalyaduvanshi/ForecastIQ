import streamlit as st
import pandas as pd
import joblib


st.set_page_config(page_title="Customer Intelligence Platform", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3 {
        color: #00ADB5;
    }
    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)


model = joblib.load("churn_model.pkl") 
columns = joblib.load("columns.pkl") 

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv") 


st.title("Customer Intelligence Platform")
st.subheader("Developed by Sonal Yaduvanshi | B.Tech CSE, PSIT Kanpur")
st.write("Analyze customer behavior & predict churn using AI")


st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Dashboard", "Data", "Prediction"])


if option == "Dashboard":
    st.header("Business Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churned Customers", df[df['Churn']=='Yes'].shape[0])
    col3.metric("Churn Rate", str(round(df[df['Churn']=='Yes'].shape[0]/len(df)*100,2)) + "%")

    st.markdown("---")

    st.subheader("Churn Distribution")
    st.bar_chart(df['Churn'].value_counts())

    st.subheader("Churn by Contract Type")
    contract = df.groupby('Contract')['Churn'].value_counts().unstack()
    st.bar_chart(contract)

    st.subheader("Monthly Charges Distribution")
    st.bar_chart(df['MonthlyCharges'])

    st.markdown("---")

    st.subheader("Key Insights")
    st.write("• Month-to-month customers churn more")
    st.write("• Higher monthly charges increase churn risk")
    st.write("• Fiber optic users show higher churn")

# ------------------ DATA ------------------
elif option == "Data":
    st.header("Dataset Preview")
    st.write("Dataset Shape:", df.shape)
    st.dataframe(df.head(50))

# ------------------ PREDICTION ------------------
elif option == "Prediction":
    st.header("Customer Churn Prediction")

    tenure = st.slider("Tenure (Months)", 0, 72)
    monthly = st.number_input("Monthly Charges", 0)

    if st.button("Predict"):
        input_df = pd.DataFrame({
            "tenure": [tenure],
            "MonthlyCharges": [monthly]
        })

        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=columns, fill_value=0)

        pred = model.predict(input_df)

        if pred[0] == 1:
            st.error("High Risk: Customer will churn")
        else:
            st.success("Safe: Customer will stay")

        st.write("Model Accuracy: ~85%")
        st.download_button("Download Dataset", df.to_csv())

# ------------------ FOOTER ------------------
st.markdown("---")

st.markdown(
    "Developed by **Sonal Yaduvanshi**  \n"
    "B.Tech CSE, PSIT Kanpur  \n"
    "Email: [sonalyaduvanshi.2k25@gmail.com](mailto:sonalyaduvanshi.2k25@gmail.com)  \n"
    "GitHub: [github.com/sonalyaduvanshi](https://github.com/sonalyaduvanshi)  \n"
    "LinkedIn: [linkedin.com/in/sonal2311](https://www.linkedin.com/in/sonal2311)"
)

import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


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
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)


with st.spinner(" Loading AI Model..."):
    model = joblib.load("churn_model.pkl")
    columns = joblib.load("columns.pkl")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.success(" App Loaded Successfully!")

st.title("Customer Intelligence Platform")

st.markdown("""
### Predict Customer Churn with AI

This platform helps businesses:
- Identify high-risk customers
- Improve retention strategies
- Make data-driven decisions
""")


st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Dashboard", "Data", "Prediction"])


if option == "Dashboard":
    st.header("Business Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", len(df))
    col2.metric("Churned Customers", df[df['Churn']=='Yes'].shape[0])
    col3.metric("Churn Rate", str(round(df[df['Churn']=='Yes'].shape[0]/len(df)*100,2)) + "%")

    st.markdown("---")

    st.subheader(" Churn Distribution")
    st.bar_chart(df['Churn'].value_counts())

    st.subheader("Churn by Contract Type")
    contract = df.groupby('Contract')['Churn'].value_counts().unstack()
    st.bar_chart(contract)

    st.subheader("Monthly Charges Distribution")
    st.line_chart(df['MonthlyCharges'])

    st.markdown("---")

    st.subheader(" Key Insights")
    st.info("""
    • Month-to-month customers churn more  
    • Higher monthly charges increase churn risk  
    • Fiber optic users show higher churn  
    """)

elif option == "Data":
    st.header(" Dataset Preview")

    st.write("Shape of Dataset:", df.shape)
    st.dataframe(df.head(50))

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Dataset",
        data=csv,
        file_name="customer_data.csv",
        mime='text/csv'
    )


elif option == "Prediction":
    st.header(" Customer Churn Prediction")

    st.info("Fill details below to predict customer churn")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72)

    with col2:
        monthly = st.number_input("Monthly Charges", 0)

    if st.button("Predict Churn"):
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

        st.markdown("### Model Performance")
        st.write("Accuracy: ~85%")


st.markdown("---")

st.markdown("""
 **Developed by Sonal Yaduvanshi**  
B.Tech CSE, PSIT Kanpur  

Email: sonalyaduvanshi.2k25@gmail.com  
GitHub: https://github.com/sonalyaduvanshi  
LinkedIn: https://www.linkedin.com/in/sonal2311
""")

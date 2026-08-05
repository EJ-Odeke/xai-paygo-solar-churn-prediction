import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import os


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PAYGo Solar Churn Dashboard",
    page_icon="☀️",
    layout="wide"
)



# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/customer_predictions.csv"
    )

    return df



df = load_data()



# =====================================================
# COMPANY LOGO + HEADER
# =====================================================


logo_path = "images/logo.png"


header_col1, header_col2 = st.columns([1,5])


with header_col1:

    if os.path.exists(logo_path):

        logo = Image.open(logo_path)

        st.image(
            logo,
            width=120
        )


with header_col2:

    st.title(
        "PAYGo Solar Customer Churn Intelligence Dashboard"
    )



# =====================================================
# DATA VALIDATION
# =====================================================


# predicted churn

if "predicted_churn" not in df.columns:

    df["predicted_churn"] = (
        df["churn_probability"]
        >=0.5
    ).astype(int)



# risk category

if "risk_category" not in df.columns:

    df["risk_category"] = pd.cut(

        df["churn_probability"],

        bins=[
            0,
            0.3,
            0.7,
            1
        ],

        labels=[
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ]

    )



# risk score

if "risk_score" not in df.columns:

    df["risk_score"] = (
        df["churn_probability"]*100
    ).round(2)



# =====================================================
# EXECUTIVE KPIs
# =====================================================


st.divider()


st.subheader(
    "Executive Summary"
)



total_customers = len(df)



high_risk = len(

    df[
        df["risk_category"]
        ==
        "High Risk"
    ]

)



medium_risk = len(

    df[
        df["risk_category"]
        ==
        "Medium Risk"
    ]

)



predicted_churn_rate = (

    df["predicted_churn"]
    .mean()*100

)



average_risk_score = (

    df["risk_score"]
    .mean()

)



col1,col2,col3,col4 = st.columns(4)



col1.metric(

    "Total Customers",

    f"{total_customers:,}"

)



col2.metric(

    "High Risk Customers",

    f"{high_risk:,}"

)



col3.metric(

    "Medium Risk Customers",

    f"{medium_risk:,}"

)



col4.metric(

    "Average Risk Score",

    f"{average_risk_score:.1f}%"

)



# =====================================================
# RISK DISTRIBUTION
# =====================================================


st.divider()


left,right = st.columns(2)



with left:


    st.subheader(
        "Customer Risk Distribution"
    )


    risk_summary = (

        df["risk_category"]

        .value_counts()

        .reset_index()

    )


    risk_summary.columns=[

        "Risk Category",

        "Customers"

    ]


    fig = px.bar(

        risk_summary,

        x="Risk Category",

        y="Customers",

        text="Customers",

        title="Risk Segmentation"

    )


    st.plotly_chart(

        fig,

        use_container_width=True

    )




with right:


    st.subheader(

        "Churn Probability Distribution"

    )


    fig2 = px.histogram(

        df,

        x="churn_probability",

        nbins=30,

        title="Predicted Churn Probability"

    )


    st.plotly_chart(

        fig2,

        use_container_width=True

    )



# =====================================================
# TOP RISK CUSTOMERS
# =====================================================


st.divider()


st.subheader(

    "🚨 Highest Risk Customers"

)



top_columns=[

    "customer_id",

    "risk_category",

    "risk_score",

    "churn_probability"

]



extra_columns=[

    "churn_driver_explanation",

    "recommended_action",

    "top_shap_features"

]



for col in extra_columns:

    if col in df.columns:

        top_columns.append(col)



top_customers=(

    df

    .sort_values(

        "churn_probability",

        ascending=False

    )

    .head(20)

)



st.dataframe(

    top_customers[top_columns],

    use_container_width=True

)



# =====================================================
# BUSINESS INTERPRETATION
# =====================================================


st.divider()


st.subheader(

    "Retention Priority Summary"

)



st.markdown(

f"""

### Key Operational Insight

- **{high_risk:,} customers** are classified as high churn risk.
- These customers should receive proactive retention intervention.
- Priority actions include:

    - Payment recovery support
    - Complaint resolution
    - Customer engagement campaigns
    - Repayment plan discussions

The dashboard supports data-driven retention decisions using
Explainable AI predictions.

"""

)
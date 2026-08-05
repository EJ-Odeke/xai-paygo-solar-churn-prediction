import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="SHAP Explainability",
    page_icon="🔍",
    layout="wide"
)


st.title("🔍 SHAP Explainability Analysis")



# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():

    dashboard_file = "data/dashboard_data.csv"
    shap_file = "data/shap_importance.csv"


    if not os.path.exists(dashboard_file):

        st.error(
            "Missing data/dashboard_data.csv"
        )

        st.stop()



    if not os.path.exists(shap_file):

        st.error(
            "Missing data/shap_importance.csv"
        )

        st.stop()



    dashboard_df = pd.read_csv(
        dashboard_file
    )


    shap_df = pd.read_csv(
        shap_file
    )


    return dashboard_df, shap_df



df, shap_importance = load_data()



# ===============================
# CHECK SHAP FILE
# ===============================

st.subheader("SHAP File Structure")


st.write(
    shap_importance.head()
)


st.write(
    "Columns:",
    list(shap_importance.columns)
)



# ===============================
# IDENTIFY SHAP VALUE COLUMN
# ===============================

importance_column = None


for column in shap_importance.columns:


    if column.lower() in [

        "mean_abs_shap",
        "importance",
        "shap_importance",
        "mean(|shap|)",
        "mean_abs",
        "value"

    ]:

        importance_column = column

        break



# If not found, use second numeric column

if importance_column is None:


    numeric_columns = (

        shap_importance
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()

    )


    if len(numeric_columns) > 0:


        importance_column = numeric_columns[0]



# ===============================
# GLOBAL SHAP IMPORTANCE
# ===============================


st.divider()

st.subheader(
    "Global Feature Importance"
)



if importance_column is None:


    st.error(
        """
        Cannot identify SHAP importance values.

        Please recreate shap_importance.csv
        from the notebook.
        """
    )

    st.stop()



# Sort SHAP importance

shap_plot = (

    shap_importance

    .sort_values(

        importance_column,

        ascending=True

    )

    .head(15)

)



fig, ax = plt.subplots(
    figsize=(8,6)
)



ax.barh(

    shap_plot.iloc[:,0],

    shap_plot[importance_column]

)



ax.set_xlabel(
    "Mean Absolute SHAP Value"
)



ax.set_ylabel(
    "Feature"
)



st.pyplot(fig)



# ===============================
# TOP FEATURES TABLE
# ===============================


st.subheader(
    "Top Churn Influencing Features"
)



top_features = (

    shap_importance

    .sort_values(

        importance_column,

        ascending=False

    )

    .head(15)

)



st.dataframe(

    top_features,

    use_container_width=True

)



# ===============================
# CUSTOMER EXPLANATION
# ===============================


st.divider()


st.subheader(
    "Customer-Level Explanation"
)



if "customer_id" not in df.columns:


    st.error(
        "customer_id missing from dashboard_data.csv"
    )

    st.stop()



selected_customer = st.selectbox(

    "Select Customer",

    df["customer_id"].unique()

)



customer = df[
    df["customer_id"] == selected_customer
].iloc[0]



col1,col2,col3 = st.columns(3)



with col1:

    st.metric(

        "Risk Level",

        customer.get(
            "risk_level",
            "Unknown"
        )

    )



with col2:

    st.metric(

        "Churn Probability",

        f"{customer.get('churn_probability',0):.2%}"

    )



with col3:

    prediction = customer.get(
        "predicted_churn",
        0
    )


    st.metric(

        "Prediction",

        "Churn" if prediction==1 else "Active"

    )



# ===============================
# CUSTOMER BEHAVIOUR
# ===============================


st.subheader(
    "Customer Behaviour Indicators"
)



metrics = [

    "repayment_rate",

    "missed_payments",

    "outstanding_balance",

    "contact_intensity",

    "payment_momentum",

    "payment_recovery_score",

    "complaint_pressure",

    "support_escalation_score"

]



available_metrics = [

    x for x in metrics

    if x in df.columns

]



profile = pd.DataFrame({

    "Metric":

        available_metrics,


    "Value":

        [

            customer[x]

            for x in available_metrics

        ]

})



st.dataframe(

    profile,

    hide_index=True,

    use_container_width=True

)



# ===============================
# RETENTION RECOMMENDATION
# ===============================


st.subheader(
    "Recommended Retention Action"
)



risk = customer.get(
    "risk_level",
    ""
)



if risk=="High Risk":


    st.error(
        """
        Actions:

        • Contact customer immediately

        • Investigate payment challenges

        • Provide repayment flexibility

        • Resolve complaints/support issues

        """

    )


elif risk=="Medium Risk":


    st.warning(
        """
        Actions:

        • Increase engagement

        • Monitor payment behaviour

        • Send reminders

        """

    )


else:


    st.success(
        """
        Actions:

        • Maintain current relationship

        • Continue normal support

        """

    )
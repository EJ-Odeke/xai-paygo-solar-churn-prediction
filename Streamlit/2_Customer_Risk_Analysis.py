import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Customer Risk Analysis",
    layout="wide"
)


# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/dashboard_data.csv"
    )

    return df


df = load_data()


# ==============================
# TITLE
# ==============================

st.title("👤 Customer Risk Analysis")

st.markdown(
"""
This section provides customer-level churn risk analysis.
Customers can be filtered by risk category and analysed using
behavioural indicators, churn explanations and recommended interventions.
"""
)



# ==============================
# FILTERS
# ==============================

col1, col2 = st.columns(2)


with col1:

    risk_filter = st.multiselect(
        "Select Risk Level",
        options=df["risk_level"].unique(),
        default=df["risk_level"].unique()
    )


with col2:

    customer_search = st.text_input(
        "Search Customer ID"
    )



filtered_df = df[
    df["risk_level"].isin(risk_filter)
]


if customer_search:

    filtered_df = filtered_df[
        filtered_df["customer_id"]
        .astype(str)
        .str.contains(customer_search)
    ]



# ==============================
# CUSTOMER SELECTION
# ==============================


customer_selected = st.selectbox(
    "Select Customer",
    filtered_df["customer_id"].tolist()
)


customer = filtered_df[
    filtered_df["customer_id"] == customer_selected
].iloc[0]



# ==============================
# RISK SUMMARY
# ==============================


st.subheader("Customer Risk Profile")


c1,c2,c3,c4 = st.columns(4)


with c1:

    st.metric(
        "Churn Probability",
        f"{customer['churn_probability']:.1%}"
    )


with c2:

    st.metric(
        "Risk Level",
        customer["risk_level"]
    )


with c3:

    st.metric(
        "Actual Churn",
        "Yes" if customer["actual_churn"]==1 else "No"
    )


with c4:

    st.metric(
        "Customer Value",
        round(customer.get(
            "customer_value",
            0
        ))
    )



st.divider()



# ==============================
# CUSTOMER BEHAVIOUR
# ==============================


st.subheader(
"📊 Customer Behaviour Indicators"
)


behaviour_features = {

"Repayment Rate":
"repayment_rate",

"Missed Payment Rate":
"missed_payment_rate",

"Payment Momentum":
"payment_momentum",

"Outstanding Balance Ratio":
"outstanding_balance_ratio",

"Payment Stability Score":
"payment_stability_score",

"Contact Intensity":
"contact_intensity",

"Complaint Pressure":
"complaint_pressure",

"Support Escalation Score":
"support_escalation_score",

"Customer Value":
"customer_value"

}



behaviour_df = pd.DataFrame(
{
"Metric":
behaviour_features.keys(),

"Value":
[
customer.get(x,np.nan)
for x in behaviour_features.values()
]

}
)



st.dataframe(
    behaviour_df,
    use_container_width=True,
    hide_index=True
)




st.divider()



# ==============================
# EXPLANATION ENGINE
# ==============================


st.subheader(
"🔍 Why is this customer at risk?"
)



feature_meanings = {

"missed_payments":
"Frequent missed payments indicating repayment difficulty",

"repayment_rate":
"Low repayment consistency",

"outstanding_balance":
"High outstanding balance increasing churn exposure",

"contact_intensity":
"Reduced customer engagement",

"complaint_pressure":
"High complaint activity indicating dissatisfaction",

"payment_momentum":
"Declining payment momentum",

"support_escalation_score":
"Unresolved support issues",

"payment_stability_score":
"Unstable payment behaviour"

}




explanations=[]


for feature,meaning in feature_meanings.items():

    if feature in customer.index:

        value = customer[feature]

        if pd.notna(value):

            if feature in [
                "missed_payments",
                "outstanding_balance",
                "complaint_pressure"
            ]:

                if value > df[feature].median():

                    explanations.append(
                        f"⚠️ {meaning}"
                    )


            elif feature in [
                "repayment_rate",
                "payment_stability_score"
            ]:

                if value < df[feature].median():

                    explanations.append(
                        f"⚠️ {meaning}"
                    )


            else:

                explanations.append(
                    f"⚠️ {meaning}"
                )



if explanations:

    for e in explanations:

        st.warning(e)

else:

    st.success(
        "No major behavioural risk drivers detected."
    )





# ==============================
# RECOMMENDATIONS
# ==============================


st.subheader(
"💡 Recommended Intervention"
)



actions=[]


if customer["missed_payments"] > df["missed_payments"].median():

    actions.append(
        "Contact customer and discuss flexible repayment options."
    )


if customer["complaint_pressure"] > df["complaint_pressure"].median():

    actions.append(
        "Prioritize complaint resolution."
    )


if customer["contact_intensity"] < df["contact_intensity"].median():

    actions.append(
        "Increase proactive customer engagement."
    )


if customer["outstanding_balance"] > df["outstanding_balance"].median():

    actions.append(
        "Create repayment recovery plan."
    )



if actions:

    for a in actions:

        st.info(a)

else:

    st.success(
        "Maintain current customer engagement strategy."
    )



# ==============================
# RAW CUSTOMER RECORD
# ==============================


with st.expander(
"View Complete Customer Record"
):

    st.dataframe(
        customer.to_frame(),
        use_container_width=True
    )
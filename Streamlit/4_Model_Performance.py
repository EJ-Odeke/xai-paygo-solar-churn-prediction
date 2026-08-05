import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Model Performance Evaluation")

st.markdown(
    """
    This section presents the evaluation results of the machine learning models
    developed for PAYGo solar customer churn prediction.
    """
)



# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_results():

    files = {

        "model_results":
        "data/model_results.csv",

        "confusion_matrix":
        "data/confusion_matrix.csv",

        "feature_importance":
        "data/feature_importance.csv"

    }


    data = {}


    for name,path in files.items():

        if os.path.exists(path):

            data[name] = pd.read_csv(path)

        else:

            data[name] = None


    return data



data = load_results()



# ===============================
# MODEL COMPARISON
# ===============================


st.subheader(
    "Model Comparison"
)



if data["model_results"] is not None:


    results = data["model_results"]


    st.dataframe(

        results,

        use_container_width=True

    )



    # Find accuracy column

    metric_column = None


    for col in results.columns:


        if col.lower() in [

            "accuracy",
            "roc_auc",
            "f1_score",
            "f1"

        ]:

            metric_column = col

            break



    if metric_column:


        fig,ax = plt.subplots(
            figsize=(8,5)
        )


        ax.bar(

            results["Model"],

            results[metric_column]

        )


        ax.set_ylabel(
            metric_column
        )


        ax.set_title(
            "Model Performance Comparison"
        )


        plt.xticks(
            rotation=45
        )


        st.pyplot(fig)



else:


    st.warning(
        """
        model_results.csv not found.

        Export model evaluation results from your notebook first.
        """
    )



st.divider()



# ===============================
# CONFUSION MATRIX
# ===============================


st.subheader(
    "Confusion Matrix"
)



if data["confusion_matrix"] is not None:


    cm = data["confusion_matrix"]


    st.dataframe(

        cm,

        use_container_width=True

    )



    if cm.shape[0] == 2:


        fig,ax = plt.subplots(
            figsize=(5,4)
        )


        ax.imshow(
            cm.values
        )


        ax.set_xticks(
            range(len(cm.columns))
        )


        ax.set_xticklabels(
            cm.columns
        )


        ax.set_yticks(
            range(len(cm.index))
        )


        ax.set_yticklabels(
            cm.index
        )


        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                ax.text(
                    j,
                    i,
                    cm.iloc[i,j],
                    ha="center",
                    va="center"
                )


        ax.set_xlabel(
            "Predicted"
        )


        ax.set_ylabel(
            "Actual"
        )


        st.pyplot(fig)



else:


    st.info(
        "Confusion matrix file not available."
    )



st.divider()



# ===============================
# FEATURE IMPORTANCE
# ===============================


st.subheader(
    "Model Feature Importance"
)



if data["feature_importance"] is not None:


    importance = data["feature_importance"]


    st.dataframe(

        importance.head(15),

        use_container_width=True

    )


    value_column = None


    for col in importance.columns:


        if col.lower() not in [

            "feature",
            "name"

        ]:

            value_column = col

            break



    if value_column:


        plot_data = (

            importance

            .sort_values(
                value_column,
                ascending=True
            )

            .head(15)

        )


        fig,ax = plt.subplots(
            figsize=(8,6)
        )


        ax.barh(

            plot_data.iloc[:,0],

            plot_data[value_column]

        )


        ax.set_xlabel(
            "Importance"
        )


        st.pyplot(fig)



else:


    st.info(
        "Feature importance file not available."
    )



# ===============================
# BEST MODEL SUMMARY
# ===============================


st.divider()


st.subheader(
    "Best Model Summary"
)


st.success(
    """
    Best performing model:

    Tuned XGBoost


    Reported performance:

    • Accuracy: 85.34%

    • Precision: 91.03%

    • Recall: 82.06%

    • ROC-AUC: 92.93%


    The model was selected because it provided the best balance
    between predictive performance and explainability.
    """
)
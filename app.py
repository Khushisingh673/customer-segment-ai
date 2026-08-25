import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px



st.set_page_config(
    page_title="Customer Segment AI",
    page_icon="🧠",
    layout="wide"
)



st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 25px;
    }

    .section-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 12px;
    }

    .result-box {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
cluster_names = joblib.load("cluster_names.pkl")
feature_names = joblib.load("feature_names.pkl")

df = pd.read_csv("data/segmented_customers.csv")



income_column = None
spending_column = None

for col in df.columns:
    if "income" in col.lower():
        income_column = col

    if "spending" in col.lower():
        spending_column = col



st.markdown(
    '<div class="main-title">🧠 Customer Segment AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Smart customer segmentation using Unsupervised Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)



with st.sidebar:
    st.header("Project Details")

    st.write("**Learning Type**")
    st.success("Unsupervised Learning")

    st.write("**Algorithm**")
    st.info("K-Means Clustering")

    st.write("**Main Features**")
    st.write("• Annual Income")
    st.write("• Spending Score")

    st.divider()

    st.write("**Tech Stack**")
    st.write("Python")
    st.write("Pandas")
    st.write("Scikit-learn")
    st.write("Streamlit")
    st.write("Plotly")



col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Customers",
    len(df)
)

col2.metric(
    "Customer Segments",
    df["Segment"].nunique()
)

col3.metric(
    "Average Spending Score",
    round(df[spending_column].mean(), 1)
)

st.divider()



tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Segment Predictor",
        "📊 Analytics",
        "📂 Dataset",
        "🧠 Model Details"
    ]
)



with tab1:

    st.subheader("Customer Segment Predictor")

    st.write(
        "Enter customer information to identify the most suitable customer segment."
    )

    left, right = st.columns(2)

    with left:
        income = st.slider(
            "Annual Income",
            min_value=int(df[income_column].min()),
            max_value=int(df[income_column].max()),
            value=int(df[income_column].mean())
        )

    with right:
        spending = st.slider(
            "Spending Score",
            min_value=1,
            max_value=100,
            value=50
        )

    if st.button(
        "🔍 Analyze Customer",
        use_container_width=True
    ):

        
        input_df = pd.DataFrame(
            [[income, spending]],
            columns=feature_names
        )

        scaled_input = scaler.transform(input_df)

        cluster = model.predict(
            scaled_input
        )[0]

        segment = cluster_names[cluster]

        st.markdown(
            f"""
            <div class="result-box">
            <h3>Customer Segment</h3>
            <h2>{segment}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Annual Income",
            income
        )

        col2.metric(
            "Spending Score",
            spending
        )

        st.subheader("Recommended Marketing Strategy")

        if segment == "Premium Customers":

            st.success(
                """
                Offer premium products,
                VIP memberships,
                loyalty rewards
                and early access to exclusive offers.
                """
            )

        elif segment == "Potential Customers":

            st.info(
                """
                Use personalized advertisements,
                special promotions
                and product recommendations
                to increase engagement.
                """
            )

        elif segment == "Budget Customers":

            st.warning(
                """
                Promote discounts,
                cashback,
                combo offers
                and value-for-money products.
                """
            )

        elif segment == "Active Customers":

            st.success(
                """
                Introduce reward points,
                customer retention programs
                and loyalty benefits.
                """
            )

        else:

            st.info(
                """
                Maintain regular engagement
                through seasonal campaigns,
                offers and personalized recommendations.
                """
            )



with tab2:

    st.subheader("Customer Analytics")

    
    scatter = px.scatter(
        df,
        x=income_column,
        y=spending_column,
        color="Segment",
        hover_data=[
            col for col in df.columns
            if col not in ["Cluster"]
        ],
        title="Customer Segmentation Map"
    )

    st.plotly_chart(
        scatter,
        use_container_width=True
    )

    # Segment Distribution
    segment_counts = (
        df["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Customers"
    ]

    bar = px.bar(
        segment_counts,
        x="Segment",
        y="Customers",
        title="Number of Customers in Each Segment"
    )

    st.plotly_chart(
        bar,
        use_container_width=True
    )

    
    st.subheader("Segment Summary")

    summary = (
        df.groupby("Segment")[
            [
                income_column,
                spending_column
            ]
        ]
        .agg(
            [
                "count",
                "mean"
            ]
        )
        .round(2)
    )

    st.dataframe(
        summary,
        use_container_width=True
    )



with tab3:

    st.subheader("Processed Dataset")

    st.write(
        f"Dataset contains **{len(df)} customer records**."
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Basic Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )



with tab4:

    st.subheader("Model Information")

    st.markdown(
        """
        ### Machine Learning Type
        **Unsupervised Learning**

        The dataset does not contain predefined customer segment labels,
        so the model discovers patterns automatically.

        ### Algorithm
        **K-Means Clustering**

        K-Means groups similar customers based on their distance from
        cluster centroids.

        ### Features Used
        - Annual Income
        - Spending Score

        ### Preprocessing
        `StandardScaler` is used to standardize numerical values.

        ### Number of Clusters
        The project uses **5 customer clusters**.

        ### Cluster Evaluation
        Two techniques were used:

        - Elbow Method
        - Silhouette Score

        ### Final Output
        Mathematical clusters are converted into understandable customer
        segments for business interpretation.
        """
    )

    st.subheader("Project Workflow")

    st.code(
        """
Customer Dataset
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
StandardScaler
        ↓
Elbow Method
        ↓
Silhouette Score
        ↓
K-Means Clustering
        ↓
Customer Segments
        ↓
Streamlit Dashboard
        ↓
Marketing Recommendation
        """
    )


st.divider()

st.caption(
    "Customer Segment AI • K-Means Clustering • Unsupervised Machine Learning"
)
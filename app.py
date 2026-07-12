import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    MeanShift
)

from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    r2_score
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Machine Learning Model Explorer",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Machine Learning Model Explorer")

st.caption(
    "Interactive Dashboard for Supervised and Unsupervised Machine Learning Algorithms"
)

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙️ Model Selection")

st.sidebar.write(
    "Choose an algorithm to train and evaluate."
)

algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    [
        "Choose Algorithm",
        "Linear Regression",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN",
        "K-Means",
        "DBSCAN",
        "Hierarchical Clustering",
        "PCA",
        "Mean Shift"
    ]
)

st.sidebar.divider()

st.sidebar.success(
    "10 Machine Learning Algorithms"
)

# ==========================================================
# HOME PAGE
# ==========================================================

if algorithm == "Choose Algorithm":

    st.header("👋 Welcome")

    st.write(
        """
This dashboard demonstrates commonly used Machine Learning Algorithms.

### Features

- Dataset Preview
- Model Training
- Performance Evaluation
- Prediction

Select an algorithm from the sidebar to begin.
"""
    )

# ==========================================================
# LINEAR REGRESSION
# ==========================================================

elif algorithm == "Linear Regression":

    data = pd.read_csv(
        "datasets/students.csv"
    )

    X = data[["Hours_Studied"]]
    y = data["Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    st.header("📈 Linear Regression")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Model Performance")

        st.metric(
            "R² Score",
            round(
                r2_score(
                    y_test,
                    y_pred
                ),
                2
            )
        )

    st.divider()

    st.subheader("🎯 Predict Student Marks")

    hours = st.number_input(
        "Study Hours",
        min_value=0.0,
        max_value=20.0,
        value=5.0
    )

    prediction = model.predict(
        [[hours]]
    )

    st.success(
        f"Predicted Marks : {prediction[0]:.2f}"
    )

# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

elif algorithm == "Logistic Regression":

    data = pd.read_csv(
        "datasets/customer_purchase.csv"
    )

    X = data[
        [
            "Age",
            "Salary"
        ]
    ]

    y = data["Purchased"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression()

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    st.header("🧮 Logistic Regression")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Model Performance")

        st.metric(
            "Accuracy",
            round(
                accuracy_score(
                    y_test,
                    y_pred
                ),
                2
            )
        )

    st.divider()

    st.subheader("🎯 Purchase Prediction")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    salary = st.number_input(
        "Salary",
        min_value=10000,
        max_value=100000,
        value=50000
    )

    prediction = model.predict(
        [[age, salary]]
    )

    if prediction[0] == 1:

        st.success(
            "Customer Will Purchase the Product"
        )

    else:

        st.error(
            "Customer Will NOT Purchase the Product"
        )

# ==========================================================
# DECISION TREE
# ==========================================================

elif algorithm == "Decision Tree":

    data = pd.read_csv(
        "datasets/loan_approval.csv"
    )

    X = data[
        [
            "Age",
            "Income",
            "Credit_Score"
        ]
    ]

    y = data["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    st.header("🌳 Decision Tree")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Model Performance")

        st.metric(
            "Accuracy",
            round(
                accuracy_score(
                    y_test,
                    y_pred
                ),
                2
            )
        )

    st.divider()

    st.subheader("🎯 Loan Approval Prediction")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=80,
        value=30
    )

    income = st.number_input(
        "Income",
        min_value=10000,
        max_value=200000,
        value=50000
    )

    credit = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700
    )

    prediction = model.predict(
        [[
            age,
            income,
            credit
        ]]
    )

    if prediction[0] == 1:

        st.success(
            "✅ Loan Approved"
        )

    else:

        st.error(
            "❌ Loan Rejected"
        )

# ==========================================================
# RANDOM FOREST
# ==========================================================

elif algorithm == "Random Forest":

    data = pd.read_csv(
        "datasets/employee_attrition.csv"
    )

    X = data[
        [
            "Age",
            "Experience",
            "Salary"
        ]
    ]

    y = data["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    st.header("🌲 Random Forest")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Model Performance")

        st.metric(
            "Accuracy",
            round(
                accuracy_score(
                    y_test,
                    y_pred
                ),
                2
            )
        )

    st.divider()

    st.subheader("🎯 Employee Attrition Prediction")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    experience = st.number_input(
        "Experience (Years)",
        min_value=0,
        max_value=40,
        value=5
    )

    salary = st.number_input(
        "Salary",
        min_value=10000,
        max_value=200000,
        value=50000
    )

    prediction = model.predict(
        [[
            age,
            experience,
            salary
        ]]
    )

    if prediction[0] == 1:

        st.error(
            "⚠️ Employee May Leave the Company"
        )

    else:

        st.success(
            "✅ Employee Likely to Stay"
        )

# ==========================================================
# KNN
# ==========================================================

elif algorithm == "KNN":

    data = pd.read_csv(
        "datasets/mobile_recommendation.csv"
    )

    X = data[
        [
            "Price",
            "RAM"
        ]
    ]

    y = data["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    st.header(" K-Nearest Neighbors")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Model Performance")

        st.metric(
            "Accuracy",
            round(
                accuracy_score(
                    y_test,
                    y_pred
                ),
                2
            )
        )

    st.divider()

    st.subheader("🎯 Mobile Recommendation")

    price = st.number_input(
        "Price",
        min_value=5000,
        max_value=100000,
        value=25000
    )

    ram = st.number_input(
        "RAM (GB)",
        min_value=2,
        max_value=16,
        value=8
    )

    prediction = model.predict(
        [[
            price,
            ram
        ]]
    )

    st.success(
        f"📱 Recommended Category : {prediction[0]}"
    )

# ==========================================================
# K-MEANS
# ==========================================================

elif algorithm == "K-Means":

    data = pd.read_csv(
        "datasets/customer_segmentation.csv"
    )

    X = data[
        [
            "Annual_Income",
            "Spending_Score"
        ]
    ]

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    data["Cluster"] = model.fit_predict(X)

    st.header("🧩 K-Means Clustering")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Cluster Information")

        st.metric(
            "Clusters",
            data["Cluster"].nunique()
        )

    st.divider()

    st.subheader("🎯 Customer Segmentation")

    income = st.number_input(
        "Annual Income",
        10000,
        200000,
        50000
    )

    spending = st.slider(
        "Spending Score",
        1,
        100,
        50
    )

    if st.button("Predict Cluster"):

        cluster = model.predict(
            [[income, spending]]
        )

        st.success(
            f"✅ Customer belongs to Cluster {cluster[0]}"
        )
# ==========================================================
# DBSCAN
# ==========================================================

elif algorithm == "DBSCAN":

    data = pd.read_csv(
        "datasets/fraud_detection.csv"
    )

    X = data[
        [
            "Amount",
            "Frequency"
        ]
    ]

    model = DBSCAN(
        eps=3000,
        min_samples=3
    )

    data["Cluster"] = model.fit_predict(X)

    st.header("🔍 DBSCAN")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Cluster Information")

        st.metric(
            "Clusters",
            len(data["Cluster"].unique())
        )

    st.divider()

    st.subheader("🎯 Transaction Analysis")

    amount = st.number_input(
        "Transaction Amount",
        min_value=0,
        value=5000
    )

    frequency = st.number_input(
        "Transaction Frequency",
        min_value=1,
        value=5
    )

    if amount > 50000 and frequency > 20:

        st.error(
            "⚠️ Possible Fraud Transaction"
        )

    else:

        st.success(
            "✅ Normal Transaction"
        )

# ==========================================================
# HIERARCHICAL CLUSTERING
# ==========================================================

elif algorithm == "Hierarchical Clustering":

    data = pd.read_csv(
        "datasets/mall_customers.csv"
    )

    X = data[
        [
            "Annual_Income",
            "Spending_Score"
        ]
    ]

    model = AgglomerativeClustering(
        n_clusters=3
    )

    data["Cluster"] = model.fit_predict(X)

    st.header("🏗️ Hierarchical Clustering")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Cluster Information")

        st.metric(
            "Clusters",
            len(data["Cluster"].unique())
        )

    st.divider()

    st.subheader("🎯 Customer Segment Analysis")

    income = st.number_input(
        "Annual Income",
        min_value=10000,
        max_value=200000,
        value=50000
    )

    spending = st.number_input(
        "Spending Score",
        min_value=1,
        max_value=100,
        value=50
    )

    cluster_means = data.groupby(
        "Cluster"
    )[
        [
            "Annual_Income",
            "Spending_Score"
        ]
    ].mean()

    distances = (
        (cluster_means["Annual_Income"] - income) ** 2
        +
        (cluster_means["Spending_Score"] - spending) ** 2
    )

    nearest_cluster = distances.idxmin()

    st.success(
        f"Customer belongs to Cluster {nearest_cluster}"
    )

# ==========================================================
# PCA
# ==========================================================

elif algorithm == "PCA":

    data = pd.read_csv(
        "datasets/student_performance.csv"
    )

    X = data[
        [
            "Maths",
            "Science",
            "English",
            "Computer"
        ]
    ]

    model = PCA(
        n_components=2
    )

    result = model.fit_transform(X)

    pca_df = pd.DataFrame(
        result,
        columns=[
            "PC1",
            "PC2"
        ]
    )

    st.header("📉 Principal Component Analysis")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Reduced Dataset")

        st.dataframe(
            pca_df.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 PCA Summary")

        st.metric(
            "Components",
            2
        )

    st.divider()

    st.subheader("🎯 Student Performance Analysis")

    maths = st.number_input(
        "Maths",
        0,
        100,
        80
    )

    science = st.number_input(
        "Science",
        0,
        100,
        75
    )

    english = st.number_input(
        "English",
        0,
        100,
        70
    )

    computer = st.number_input(
        "Computer",
        0,
        100,
        90
    )

    student = pd.DataFrame(
        [[
            maths,
            science,
            english,
            computer
        ]],
        columns=[
            "Maths",
            "Science",
            "English",
            "Computer"
        ]
    )

    transformed = model.transform(
        student
    )

    st.success(
        f"PC1 : {transformed[0][0]:.2f}"
    )

    st.success(
        f"PC2 : {transformed[0][1]:.2f}"
    )

# ==========================================================
# MEAN SHIFT
# ==========================================================

elif algorithm == "Mean Shift":

    data = pd.read_csv(
        "datasets/store_customers.csv"
    )

    X = data[
        [
            "Monthly_Spending",
            "Visits_Per_Month"
        ]
    ]

    model = MeanShift()

    data["Cluster"] = model.fit_predict(X)

    st.header("🎯 Mean Shift Clustering")

    col1, col2 = st.columns([2,1])

    with col1:

        st.subheader("📄 Dataset Preview")

        st.dataframe(
            data.head(10),
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Cluster Information")

        st.metric(
            "Clusters Found",
            len(
                data["Cluster"].unique()
            )
        )

    st.divider()

    st.subheader("🎯 Customer Cluster Prediction")

    spending = st.number_input(
        "Monthly Spending",
        min_value=1000,
        max_value=100000,
        value=15000
    )

    visits = st.number_input(
        "Visits Per Month",
        min_value=1,
        max_value=50,
        value=5
    )

    centers = model.cluster_centers_

    distances = []

    for center in centers:

        distance = (
            (center[0] - spending) ** 2
            +
            (center[1] - visits) ** 2
        )

        distances.append(distance)

    cluster = distances.index(
        min(distances)
    )

    st.success(
        f"Customer belongs to Cluster {cluster}"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Machine Learning Model Explorer | "
    "Built using Python, Streamlit and Scikit-Learn"
)
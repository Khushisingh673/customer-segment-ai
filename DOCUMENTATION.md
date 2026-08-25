# 🧠 Customer Segment AI

## Project Documentation

**Project Name:** Customer Segment AI
**Domain:** Machine Learning / Customer Analytics
**Machine Learning Type:** Unsupervised Learning
**Algorithm:** K-Means Clustering
**Frontend/UI:** Streamlit
**Programming Language:** Python

---

# 1. Introduction

Customer Segment AI is a machine learning-based application designed to automatically group customers according to similarities in their purchasing behavior.

Businesses interact with customers who have different incomes, spending habits, preferences, and purchasing capacities. Treating every customer in exactly the same way may lead to ineffective marketing.

Customer segmentation helps divide customers into meaningful groups so that different strategies can be designed for different types of customers.

This project uses **K-Means Clustering**, an unsupervised machine learning algorithm, to discover customer groups based primarily on:

* Annual Income
* Spending Score

An interactive **Streamlit dashboard** is also provided so that users can explore customer data, visualize clusters, enter new customer information, and obtain a customer segment with a suggested marketing strategy.

---

# 2. Problem Statement

Businesses may have hundreds or thousands of customers with different purchasing behaviors.

For example:

* Some customers have high income and spend heavily.
* Some have high income but spend less.
* Some customers are price-sensitive.
* Some have lower income but actively purchase products.
* Some show average income and average spending behavior.

Using the same marketing strategy for all these customers may not produce good results.

Therefore, the problem addressed by this project is:

> **How can machine learning automatically discover groups of customers with similar income and spending behavior so that businesses can better understand and target their customers?**

Customer Segment AI addresses this problem using clustering.

---

# 3. Project Objectives

The main objectives of the project are:

1. Analyze customer data and purchasing behavior.
2. Understand the concept of unsupervised machine learning.
3. Apply K-Means clustering to real-world customer data.
4. Identify meaningful groups of customers.
5. Perform data preprocessing before model training.
6. Determine an appropriate number of clusters.
7. Evaluate the quality of clustering.
8. Visualize customer groups using interactive graphs.
9. Build an easy-to-use Streamlit interface.
10. Provide simple marketing recommendations for different customer groups.

---

# 4. Dataset

## 4.1 Dataset Description

The project uses a **Shopping Mall Customer Segmentation dataset** obtained from Kaggle.

The dataset contains information describing customers and their purchasing characteristics.

Depending on the exact version of the dataset, fields can include information such as:

* Customer ID
* Gender
* Age
* Annual Income
* Spending Score

For the clustering model in this project, the two primary features selected are:

### Annual Income

Annual Income represents the approximate yearly income of a customer.

It helps estimate the purchasing capacity of the customer.

### Spending Score

Spending Score represents customer spending behavior.

The score generally ranges from:

```text
1 – 100
```

A higher value represents stronger spending activity, while a lower value represents comparatively lower spending activity.

---

# 5. Why These Features Were Selected

Although the dataset contains additional customer information, this project primarily uses:

```text
Annual Income
+
Spending Score
```

These features were selected because they provide a simple representation of both:

* Purchasing capacity
* Purchasing behavior

They are also suitable for two-dimensional visualization.

For example:

```text
High Income + High Spending
            ↓
     Valuable Customer


High Income + Low Spending
            ↓
     Potential Customer


Low Income + Low Spending
            ↓
      Budget Customer
```

This makes the resulting clusters easier to understand and explain.

---

# 6. Machine Learning Type

## Unsupervised Learning

This project uses **Unsupervised Machine Learning**.

In supervised learning, the training dataset contains predefined output labels.

For example:

```text
Email → Spam

Email → Not Spam
```

The model learns the relationship between input features and known output labels.

However, in our customer dataset, we do not initially have labels such as:

```text
Customer 1 → Premium

Customer 2 → Budget

Customer 3 → Regular
```

Instead, we only have customer characteristics.

Therefore, the machine learning model must automatically discover patterns and groups within the data.

This makes customer segmentation an appropriate **unsupervised learning problem**.

---

# 7. Algorithm Used

## K-Means Clustering

The primary machine learning algorithm used in this project is:

**K-Means Clustering**

K-Means is an unsupervised machine learning algorithm that divides data points into a specified number of groups called **clusters**.

Customers with similar characteristics are placed into the same cluster.

---

# 8. How K-Means Works

Suppose we want to create `K` customer groups.

K-Means approximately follows these steps:

### Step 1 — Choose K

Select the number of clusters to create.

For example:

```text
K = 5
```

means that the algorithm will attempt to create five customer groups.

### Step 2 — Initialize Centroids

K-Means initializes center points for the clusters.

These center points are called:

**Centroids**

### Step 3 — Calculate Distance

The algorithm calculates the distance between each customer and the cluster centroids.

### Step 4 — Assign Customers

Each customer is assigned to the nearest centroid.

### Step 5 — Update Centroids

After assigning customers, K-Means recalculates the center of each cluster.

### Step 6 — Repeat

The assignment and centroid-update process continues until the cluster centers become stable.

The final result is a set of customer groups containing customers with similar characteristics.

---

# 9. Project Workflow

The overall project workflow is:

```text
Customer Dataset
        ↓
Data Exploration
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Feature Scaling
        ↓
Elbow Method
        ↓
Silhouette Score
        ↓
K-Means Training
        ↓
Customer Clusters
        ↓
Cluster Interpretation
        ↓
Save Trained Model
        ↓
Streamlit Application
        ↓
Customer Prediction
        ↓
Business Recommendation
```

---

# 10. Data Exploration

Before training the model, the dataset is explored using Pandas.

The following operations are performed:

### Display First Rows

```python
df.head()
```

This helps understand the structure of the dataset.

### Dataset Shape

```python
df.shape
```

This provides the number of rows and columns.

### Column Names

```python
df.columns
```

This identifies the available features.

### Dataset Information

```python
df.info()
```

This provides information about data types and non-null values.

### Statistical Summary

```python
df.describe()
```

This provides statistics such as:

* Mean
* Minimum
* Maximum
* Standard deviation
* Quartiles

---

# 11. Data Cleaning

Before model training, the dataset is checked for data quality problems.

## Missing Values

Missing values are checked using:

```python
df.isnull().sum()
```

Missing data can negatively affect machine learning models and therefore should be identified before training.

## Duplicate Records

Duplicate rows are checked using:

```python
df.duplicated().sum()
```

Duplicate records can unnecessarily influence the clustering results.

## Column Cleaning

Extra spaces in column names are removed using:

```python
df.columns = df.columns.str.strip()
```

This prevents errors when accessing dataset columns.

---

# 12. Feature Selection

The two selected features are:

```text
Annual Income
Spending Score
```

They are stored as the feature matrix `X`.

Conceptually:

```python
X = df[
    [
        "Annual Income",
        "Spending Score"
    ]
]
```

The exact column names depend on the downloaded dataset.

---

# 13. Feature Scaling

The project uses:

**StandardScaler**

from Scikit-learn.

Example:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

## Why Scaling Is Important

Machine learning features may have different numerical ranges.

K-Means uses distance calculations when assigning points to clusters.

Therefore, a feature with a larger numerical scale could influence the calculated distance more strongly.

StandardScaler transforms the selected features to a comparable statistical scale.

This makes the clustering process more balanced.

---

# 14. Elbow Method

One important question in K-Means is:

> How many clusters should we create?

The **Elbow Method** helps analyze this.

Multiple K-Means models are trained using different values of `K`.

For example:

```text
K = 1
K = 2
K = 3
...
K = 10
```

For each value, the model's **inertia** is calculated.

Inertia measures how close data points are to their assigned cluster centers.

The values are plotted on a graph:

```text
Inertia
   |
   |\
   | \
   |  \
   |   \__
   |      \____
   |
   +----------------
       Number of Clusters
```

The point where the reduction begins to slow significantly resembles an elbow.

That point can provide evidence for choosing an appropriate value of `K`.

The project saves this visualization as:

```text
elbow_method.png
```

---

# 15. Silhouette Score

The project also calculates the **Silhouette Score**.

The Silhouette Score evaluates how well data points fit within their assigned clusters compared with neighboring clusters.

The score generally ranges from:

```text
-1 to +1
```

Interpretation:

```text
Close to +1
→ Clusters are relatively well separated.

Close to 0
→ Clusters may overlap.

Negative
→ Some points may have been assigned poorly.
```

The project calculates silhouette scores for several possible values of `K`.

This provides an additional way to evaluate cluster structure rather than selecting `K` randomly.

---

# 16. Final Model Training

After analyzing the clustering structure, the final K-Means model is trained.

The implementation uses:

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```

The model is trained using the scaled features.

```python
clusters = kmeans.fit_predict(
    X_scaled
)
```

The resulting cluster ID is added to the dataset.

For example:

```text
Customer    Income    Spending    Cluster

C1            25         80          2
C2            85         20          3
C3            90         85          1
```

---

# 17. Understanding Cluster IDs

K-Means returns numerical cluster identifiers such as:

```text
Cluster 0
Cluster 1
Cluster 2
Cluster 3
Cluster 4
```

These numbers have no predefined business meaning.

For example:

`Cluster 0` does not automatically mean Premium Customers.

Therefore, the project examines the average income and spending characteristics of the cluster centers and converts the mathematical clusters into human-readable customer segments.

---

# 18. Customer Segments

The application interprets clusters using categories such as:

## Premium Customers

Typical characteristics:

* Higher income
* Higher spending

Possible strategy:

* Premium products
* VIP memberships
* Exclusive offers
* Loyalty rewards

---

## Potential Customers

Typical characteristics:

* Higher income
* Comparatively lower spending

These customers have purchasing capacity but may not currently spend heavily.

Possible strategy:

* Personalized promotions
* Targeted advertising
* Product recommendations
* Special offers

---

## Budget Customers

Typical characteristics:

* Lower purchasing capacity
* Lower spending

Possible strategy:

* Discounts
* Cashback
* Combo offers
* Value-for-money products

---

## Active Customers

Typical characteristics:

* Comparatively lower income
* Higher spending activity

Possible strategy:

* Loyalty programs
* Reward points
* Retention campaigns
* Personalized offers

---

## Regular Customers

Typical characteristics:

* Moderate income
* Moderate spending

Possible strategy:

* Seasonal campaigns
* Regular promotions
* Personalized recommendations
* Customer engagement programs

---

# 19. Saving the Trained Model

After training, the model is saved using **Joblib**.

This means the model does not need to be retrained every time the Streamlit application starts.

The following files are generated:

### `kmeans_model.pkl`

Contains the trained K-Means model.

### `scaler.pkl`

Contains the fitted StandardScaler.

The same scaler must be used for new customer input.

### `cluster_names.pkl`

Stores the relationship between cluster IDs and customer segment names.

### `feature_names.pkl`

Stores the feature names and their expected order.

### `segmented_customers.csv`

Contains the original customer information together with the generated cluster and segment.

---

# 20. User Interface

The frontend of the application is built using:

**Streamlit**

Streamlit allows Python-based machine learning models to be integrated into interactive web applications without requiring a separate frontend framework.

The application contains four main sections:

1. Segment Predictor
2. Analytics
3. Dataset
4. Model Details

---

# 21. Customer Segment Predictor

The Customer Segment Predictor allows a user to provide:

```text
Annual Income
Spending Score
```

The application converts the values into the expected model format.

The workflow is:

```text
User Input
     ↓
Create Feature Data
     ↓
StandardScaler
     ↓
K-Means Model
     ↓
Cluster ID
     ↓
Customer Segment
     ↓
Marketing Recommendation
```

### Application Screenshot

![Customer Segment Predictor](screenshots/Predication.png)

The output displays the identified customer segment and a suitable marketing recommendation.

---

# 22. Analytics Dashboard

The Analytics section provides an interactive representation of the customer segmentation results.

It contains information such as:

* Total customers
* Number of customer segments
* Average spending score
* Customer cluster visualization
* Customers per segment
* Segment summary

### Customer Segmentation Graph

A Plotly scatter plot is used to visualize customers.

The X-axis represents:

```text
Annual Income
```

The Y-axis represents:

```text
Spending Score
```

Customers are visually separated according to their identified segments.

### Dashboard Screenshot

![Customer Analytics Dashboard](screenshots/Dashboard.png)

---

# 23. Dataset Viewer

The Dataset section allows users to inspect the processed customer data directly from the application.

It displays information such as:

```text
Customer information
Annual Income
Spending Score
Cluster
Segment
```

Basic descriptive statistics are also displayed.

This helps users understand the data used by the machine learning system.

---

# 24. Model Information Section

The Model Details section explains the machine learning pipeline directly inside the application.

It describes:

* Unsupervised Learning
* Feature Selection
* StandardScaler
* K-Means Clustering
* Number of Clusters
* Elbow Method
* Silhouette Score
* Customer Segments

### Home Screenshot

![Home Page](screenshots/Home.png)

---

# 25. Technologies Used

## Python

Used as the main programming language.

## Pandas

Used for:

* Reading CSV files
* Data manipulation
* Dataset analysis
* Creating processed datasets

## NumPy

Used for numerical data handling.

## Scikit-learn

Used for:

* StandardScaler
* K-Means Clustering
* Silhouette Score

## Matplotlib

Used for generating the Elbow Method graph.

## Plotly

Used for interactive charts and visualizations.

## Streamlit

Used to create the interactive web application.

## Joblib

Used to save and load trained machine learning objects.

---

# 26. Project Structure

```text
Customer-Segment-AI/
│
├── data/
│   ├── Mall_Customers.csv
│   └── segmented_customers.csv
│
├── screenshots/
│   ├── predictor.png
│   ├── dashboard.png
│   └── model-details.png
│
├── app.py
├── explore.py
├── train_model.py
│
├── kmeans_model.pkl
├── scaler.pkl
├── cluster_names.pkl
├── feature_names.pkl
│
├── elbow_method.png
│
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
└── .gitignore
```

---

# 27. Role of Important Files

## `explore.py`

Used for initial dataset exploration.

It checks:

* First rows
* Dataset dimensions
* Column names
* Missing values
* Duplicates
* Descriptive statistics

## `train_model.py`

Contains the complete machine learning training pipeline.

It performs:

```text
Dataset Loading
       ↓
Data Validation
       ↓
Feature Selection
       ↓
Standardization
       ↓
Elbow Analysis
       ↓
Silhouette Analysis
       ↓
K-Means Training
       ↓
Cluster Interpretation
       ↓
Model Saving
```

## `app.py`

Contains the Streamlit application.

It loads the previously trained model and allows users to interact with it.

## `requirements.txt`

Contains the Python packages required to run the project.

## `README.md`

Provides a quick overview for GitHub visitors.

## `DOCUMENTATION.md`

Contains the detailed technical explanation of the project.

---

# 28. Installation

Clone the GitHub repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Move into the project:

```bash
cd Customer-Segment-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 29. Training the Model

Run:

```bash
python train_model.py
```

This performs the machine learning pipeline and generates files including:

```text
kmeans_model.pkl
scaler.pkl
cluster_names.pkl
feature_names.pkl
elbow_method.png
data/segmented_customers.csv
```

---

# 30. Running the Application

Run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically similar to:

```text
http://localhost:8501
```

Open it in a web browser to use the application.

---

# 31. Results

The project successfully demonstrates how unsupervised machine learning can be used to discover groups within customer data.

The system:

* Loads and validates customer data.
* Selects relevant numerical features.
* Standardizes the features.
* Applies K-Means clustering.
* Analyzes cluster quality.
* Generates customer segments.
* Visualizes customer groups.
* Accepts new customer information.
* Assigns new customers to learned clusters.
* Provides simple business recommendations.

The main goal is not to predict a predefined class but to discover meaningful structure within customer behavior.

---

# 32. Advantages

The main advantages of the project are:

### Easy to Understand

K-Means provides an intuitive introduction to unsupervised machine learning.

### Real-World Application

Customer segmentation is widely applicable in marketing and customer analytics.

### Interactive

The Streamlit interface allows users to interact directly with the machine learning model.

### Visual

Plotly graphs make customer clusters easier to understand.

### Complete ML Pipeline

The project demonstrates:

```text
Data
→ Preprocessing
→ Training
→ Evaluation
→ Saving Model
→ UI Integration
```

### Lightweight

The project can run on a normal laptop and does not require GPU hardware.

---

# 33. Limitations

Although useful as a mini-project, the current implementation has some limitations.

### Limited Features

The clustering model primarily considers annual income and spending score.

Actual customer behavior can depend on many more factors.

### Fixed Number of Clusters

The final model uses a predefined number of clusters after analysis.

Different datasets may require a different value of `K`.

### Dataset Size

The project uses an educational dataset rather than a large production customer database.

### Cluster Interpretation

Names such as Premium Customers and Budget Customers are interpretations of the discovered clusters.

They are not original ground-truth labels from the dataset.

### No Real-Time Database

Customer information is currently stored in CSV files.

---

# 34. Future Scope

The project can be improved in several ways.

## Add More Features

Future versions could include:

* Age
* Purchase frequency
* Average order value
* Product preferences
* Customer lifetime value
* Website activity

## Compare Clustering Algorithms

K-Means could be compared with:

* DBSCAN
* Hierarchical Clustering
* Gaussian Mixture Models

## Automatic Cluster Selection

The system could automatically select the most appropriate number of clusters based on evaluation metrics.

## Database Integration

Customer information could be stored using:

* MySQL
* PostgreSQL
* MongoDB

## Authentication

An authentication system could be added for business users.

## Cloud Deployment

The Streamlit application could be deployed online so that users can access it without running it locally.

## Real Business Data

A future production version could be trained using anonymized real customer transaction data.

---

# 35. Conclusion

Customer Segment AI is an end-to-end unsupervised machine learning project that demonstrates how customer data can be analyzed and automatically grouped using K-Means Clustering.

The project begins with customer data exploration and preprocessing. Annual Income and Spending Score are selected as the primary features and standardized using StandardScaler.

The Elbow Method and Silhouette Score are used to analyze the clustering structure. K-Means then identifies groups of customers with similar characteristics.

The discovered clusters are converted into understandable business categories such as Premium, Potential, Active, Regular, and Budget customers.

Finally, the trained model is integrated into a Streamlit application that provides interactive prediction, data visualization, analytics, and simple marketing recommendations.

Therefore, the project demonstrates the complete machine learning workflow:

```text
Dataset
   ↓
Exploration
   ↓
Preprocessing
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Cluster Evaluation
   ↓
Customer Segmentation
   ↓
Model Saving
   ↓
Streamlit UI
   ↓
Business Insights
```

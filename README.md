📊 App User Behavior Segmentation Dashboard
🚀 Project Overview

This project is a User Segmentation Dashboard built using Machine Learning (K-Means Clustering) and Streamlit.
It analyzes user behavior data and groups users into meaningful segments to help businesses take data-driven decisions.

🎯 Objective

To segment users based on their behavior and provide actionable insights such as:

Identifying high-value users
Detecting at-risk users
Improving user engagement
Reducing churn
🧠 Machine Learning Approach
🔹 1. Data Preprocessing
Handled missing values using median imputation
Selected relevant behavioral features
🔹 2. Feature Scaling
Used StandardScaler to normalize data
🔹 3. Clustering
Applied K-Means Clustering
Number of clusters: 4
🔹 4. User Segmentation

Clusters are mapped into meaningful categories:

🟢 High Value Users
🔵 Moderate Users
🔴 At-Risk Users
🟡 Occasional Users
📊 Dashboard Features
📌 1. Interactive Filters
Select user segments dynamically from sidebar
📌 2. KPI Metrics
Total Users
Average Engagement Score
Average Churn Risk
📌 3. User Segment Distribution
Bar chart showing number of users in each segment
Selected segment is highlighted
📌 4. Cluster Insights
Displays average behavior metrics per segment
Highlights selected segment
📌 5. Engagement vs Churn Analysis
Scatter plot showing relationship between engagement and churn
Selected segment highlighted, others faded
📌 6. Recommended Actions
Business strategies for each segment:
High Value → Loyalty rewards
Moderate → Personalized recommendations
At-Risk → Retention campaigns
Occasional → Re-engagement notifications
🛠️ Tech Stack
Python
Pandas
Scikit-learn
Matplotlib
Streamlit

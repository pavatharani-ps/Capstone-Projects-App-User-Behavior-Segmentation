import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="User Segmentation Dashboard", page_icon="🚀")

st.title("📊 App User Behavior Segmentation Dashboard")

# ---------------- DATA PROCESS FUNCTION ----------------
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("./excel/app_user_behavior_dataset.csv")

    # Fill missing values
    df["rating_given"].fillna(df["rating_given"].median(), inplace=True)

    # Feature selection
    selected_features = [
        "sessions_per_week",
        "avg_session_duration_min",
        "daily_active_minutes",
        "feature_clicks_per_session",
        "notifications_opened_per_week",
        "in_app_search_count",
        "pages_viewed_per_session",
        "days_since_last_login",
        "engagement_score",
    ]

    df_selected = df[selected_features]

    # Scaling
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_selected),
        columns=selected_features
    )

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(df_scaled)

    # Segment names
    cluster_names = {
        0: "High Value Users",
        1: "Moderate Users",
        2: "At-Risk Users",
        3: "Occasional Users",
    }

    df["user_segment"] = df["cluster"].map(cluster_names)

    # Recommended actions
    def assign_action(segment):
        if segment == "High Value Users":
            return "Offer premium plans & loyalty rewards"
        elif segment == "Moderate Users":
            return "Send personalized recommendations"
        elif segment == "At-Risk Users":
            return "Run retention campaigns & discounts"
        else:
            return "Send re-engagement notifications"

    df["recommended_action"] = df["user_segment"].apply(assign_action)

    return df


# ---------------- LOADER ----------------
with st.spinner("🔄 Loading and processing data... Please wait"):
    df = load_and_process_data()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filter Options")
selected_segment = st.sidebar.selectbox(
    "Select User Segment",
    ["All"] + list(df["user_segment"].unique())
)

# Filter data
if selected_segment != "All":
    df_filtered = df[df["user_segment"] == selected_segment]
else:
    df_filtered = df

# ---------------- KPI SECTION ----------------
st.subheader("📌 Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Total Users", len(df))
col2.metric("Avg Engagement", round(df["engagement_score"].mean(), 2))
col3.metric("Avg Churn Risk", round(df["churn_risk_score"].mean(), 2))

# ---------------- SEGMENT DISTRIBUTION ----------------
st.subheader("📊 User Segment Distribution")

fig1, ax1 = plt.subplots()

segment_counts = df["user_segment"].value_counts()

# Create colors list
colors = []
for seg in segment_counts.index:
    if selected_segment == "All":
        colors.append("skyblue")  # normal color
    elif seg == selected_segment:
        colors.append("orange")   # highlighted bar
    else:
        colors.append("lightgray")  # faded bars

# Plot bar chart with colors
segment_counts.plot(kind="bar", ax=ax1, color=colors)

ax1.set_xlabel("User Segment")
ax1.set_ylabel("Count")

st.pyplot(fig1)

# ---------------- CLUSTER INSIGHTS ----------------
st.subheader("📈 Cluster Insights (Average Values)")

cluster_profile = df.groupby("user_segment")[
    [
        "sessions_per_week",
        "avg_session_duration_min",
        "daily_active_minutes",
        "engagement_score",
        "churn_risk_score",
    ]
].mean()

# 🔥 Highlight function
def highlight_row(row):
    if selected_segment != "All" and row.name == selected_segment:
        return ["background-color: orange"] * len(row)
    elif selected_segment != "All":
        return ["background-color: lightgray"] * len(row)
    else:
        return [""] * len(row)

# Apply styling
styled_df = cluster_profile.style.apply(highlight_row, axis=1)

st.dataframe(styled_df)

# ---------------- SCATTER PLOT ----------------
st.subheader("📉 Engagement vs Churn Risk")

fig2, ax2 = plt.subplots()

if selected_segment == "All":
    # Show all segments in different colors
    for seg in df["user_segment"].unique():
        seg_data = df[df["user_segment"] == seg]
        ax2.scatter(
            seg_data["engagement_score"],
            seg_data["churn_risk_score"],
            label=seg
        )
else:
    # Separate selected vs others
    other_data = df[df["user_segment"] != selected_segment]
    selected_data = df[df["user_segment"] == selected_segment]

    # Plot other users (faded)
    ax2.scatter(
        other_data["engagement_score"],
        other_data["churn_risk_score"],
        alpha=0.2,
        label="Other Users"
    )

    # Plot selected users (highlight)
    ax2.scatter(
        selected_data["engagement_score"],
        selected_data["churn_risk_score"],
        alpha=1.0,
        label=selected_segment
    )

ax2.set_xlabel("Engagement Score")
ax2.set_ylabel("Churn Risk")
ax2.legend()

st.pyplot(fig2)
# ---------------- ACTIONS ----------------st.subheader("🎯 Recommended Actions")

actions = df.groupby("user_segment")["recommended_action"].first()

# Convert to DataFrame for styling
actions_df = actions.reset_index()

# Highlight function
def highlight_action(row):
    if selected_segment != "All" and row["user_segment"] == selected_segment:
        return ["background-color: orange"] * len(row)
    elif selected_segment != "All":
        return ["background-color: lightgray"] * len(row)
    else:
        return [""] * len(row)

# Apply style
styled_actions = actions_df.style.apply(highlight_action, axis=1)

st.write(styled_actions)
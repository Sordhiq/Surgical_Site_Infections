import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import google.generativeai as genai

# Page config
st.set_page_config(page_title="SSI Analytics App", page_icon="⚕️", layout="wide")
# Set title name
st.title("Surgical Site Infections Analytics App")
# Title banner
html_temp = """
    <div style="background-color:teal;padding:13px">
        <h2 style="color:white;text-align:center;">Byte x Brains 💻🧠</h2>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation Pane")
page = st.sidebar.radio("Go to", ["Homepage", "Dashboard Overview", "Hypothesis Testing", "Policy Recommendations"])

# File uploader
st.sidebar.markdown("### Upload Your SSI Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Load data
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("cleaned_ssi_data.csv")  # Default
    return df

df = load_data(uploaded_file)

# -----------------------------
# Homepage with App Description
# -----------------------------
if page == "Homepage":
    st.title("Welcome to the SSI Analytics App")
    st.markdown("""
    This interactive dashboard is designed to help users analyze and interpret **Surgical Site Infection (SSI)** data\
    from healthcare centers across California. It provides insights into infection trends, hospital risk factors,\
    and performance indicators to support data-driven health policy decisions.
    
    #### Features:
    - **Dashboard Overview**: Get quick metrics and visualizations.
    - **Hypothesis Testing**: Compare infection rates between hospital types.
    - **Policy Recommendations**: Generate tailored health policy ideas based on data insights.
    
    👉 Use the sidebar (>>) at the top-left corner to navigate through the app.
    """)
    st.info("Proudly developed by:")
    st.success("📌 Sodiq Jinad")

# -----------------------------
# Dashboard Overview
# -----------------------------
elif page == "Dashboard Overview" and df is not None:
    st.title("Surgical Site Infection (SSI) Analytics App")
    st.subheader("Key Metrics and Visualizations")

    st.markdown("### Summary Statistics")
    st.write(df.describe())

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Infection ratio by county
    st.markdown("### Average Top 10 Counties by Infections Reported")
    top_counties = df.groupby("County")["Infections_Reported"].mean().sort_values(ascending=True).head(10)
    st.bar_chart(top_counties)

    # SIR by procedure
    st.markdown("### Average SIR by Operative Procedure")
    avg_sir_op = df.groupby("Operative_Procedure")["SIR"].mean().sort_values(ascending=True).dropna()
    st.bar_chart(avg_sir_op)

    st.info("Proudly developed by:")
    st.success("📌 Sodiq Jinad")

# -----------------------------
# Hypothesis Testing
# -----------------------------
elif page == "Hypothesis Testing" and df is not None:
    st.title("Hypothesis Testing")

    st.markdown("### Comparing SIR between Hospitals by Bed Size")
    small_hospitals = df[df['Hospital_Category_RiskAdjustment'] == 'Smaller hospitals (<250 beds)']['SIR'].dropna()
    large_hospitals = df[df['Hospital_Category_RiskAdjustment'] == 'Larger hospitals (>=250 beds)']['SIR'].dropna()

    t_stat, p_val = stats.ttest_ind(large_hospitals, small_hospitals, equal_var=False)

    st.write(f"T-statistic: {t_stat}")
    st.write(f"P-value: {p_val}")

    if p_val < 0.05:
        st.success("There is a statistically significant difference in infection ratio between large and small-sized hospital beds (p < 0.05).")
    else:
        st.info("There is no statistically significant difference in infection ratio between large and small-sized hospital beds.")

    # Boxplot
    st.markdown("### SIR Distribution by Hospital Size")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Hospital_Category_RiskAdjustment", y="SIR", data=df)
    plt.xticks(rotation=30)
    st.pyplot(fig2)

# -----------------------------
# Policy Recommendations
# -----------------------------
elif page == "Policy Recommendations" and df is not None:
    st.title("Health Policy Recommendations")

    user_context = st.text_area("Add additional information or policy context (optional):", "")

    if st.button("Generate Recommendations"):
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)

            # Extract useful insights
            high_sir_procedures = df.groupby("Operative_Procedure")["SIR"].mean().sort_values(ascending=False).head(3)
            top_procedures_text = ", ".join(high_sir_procedures.index.tolist())

            prompt = (
                "You are a seasoned public health policy analyst. Based on the data insights below, "
                "In 300 words, generate 5 clear, simple and practical recommendations to reduce the Standardized Surgical Infection Ratio (SIR) "
                "across California hospitals:\n\n"
                f"Highest SIRs observed in procedures: {top_procedures_text}.\n\n"
                f"User Context: {user_context if user_context else 'No additional context provided.'}\n\n"
                "Keep recommendations simple, realistic, relevant, evidence-informed and avoid the use of ambigous words."
            )

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            st.markdown("### Tailored Recommendations")
            st.write(response.text)

            st.info("Proudly developed by:")
            st.success("📌 Sodiq Jinad")

        except KeyError:
            st.error("🔑 GEMINI_API_KEY not found. Please add it to your `.streamlit/secrets.toml` file.")
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")

else:
    st.warning("⚠️ Please upload a valid CSV file or ensure the dataset is accessible.")

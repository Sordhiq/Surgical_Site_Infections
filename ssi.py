import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import google.generativeai as genai

# Page configuration
st.set_page_config(page_title="California SSI Analytics App", page_icon="⚕️", layout="wide")
html_temp = """
    <div style="background-color:tomato;padding:13px">
        <h1 style="color:white;text-align:center;">Byte x Brains 💻🧠</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_ssi_data.csv")
    return df

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Hypothesis Testing", "Policy Recommendations"])

# Dashboard Overview
if page == "Dashboard Overview":
    st.title("Standardized Surgical Infection (SSI) Analytics - California")
    st.subheader("Key Metrics and Visualizations")

    # Summary stats
    st.markdown("### Summary Statistics")
    st.write(df.describe())

    # Heatmap
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Infection ratio by county
    st.markdown("### Top 10 Counties by Infections Reported")
    top_counties = df.groupby("County")["Infections_Reported"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_counties)

    # SIR by Operative Procedure
    st.markdown("### Average SIR by Operative Procedure")
    avg_sir_op = df.groupby("Operative_Procedure")["SIR"].mean().sort_values(ascending=False).dropna()
    st.bar_chart(avg_sir_op)

# Hypothesis Testing
elif page == "Hypothesis Testing":
    st.title("Hypothesis Testing")

    # T-test between large and small hospitals
    st.markdown("### Comparing SIR between Hospitals by Bed Size")
    small_hospitals = df[df['Hospital_Category_RiskAdjustment'] == 'Smaller hospitals (<250 beds)']['SIR'].dropna()
    large_hospitals = df[df['Hospital_Category_RiskAdjustment'] == 'Larger hospitals (>=250 beds)']['SIR'].dropna()

    t_stat, p_val = stats.ttest_ind(large_hospitals, small_hospitals, equal_var=False)

    st.write(f"T-statistic: {t_stat:.4f}")
    st.write(f"P-value: {p_val:.4f}")

    if p_val < 0.05:
        st.success("There is a statistically significant difference in SIR between large and small-sized hospital beds (p < 0.05).")
    else:
        st.info("No statistically significant difference in SIR between large and small-sized hospital beds.")

    # Boxplot
    st.markdown("### SIR Distribution by Hospital Size")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="Hospital_Category_RiskAdjustment", y="SIR", data=df)
    plt.xticks(rotation=30)
    st.pyplot(fig2)

# AI Recommendations
elif page == "Policy Recommendations":
    st.title("Health Policy Recommendations")

    # User input for context
    user_context = st.text_area("Provide additional context (optional):", "")

    if st.button("Generate Recommendations"):
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

        prompt = (
            "As a health policy analyst, generate 5 actionable and evidence-based policy recommendations "
            "to help reduce the Standardized Surgical Infection Ratio (SIR) across health centers in California. "
            f"Use the following context if useful:\n{user_context}"
        )

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        st.markdown("### AI Recommendations")
        st.write(response.text)

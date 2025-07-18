import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import google.generativeai as genai

# Page configuration
st.set_page_config(page_title="CSSI Analytics App", page_icon="⚕️", layout="wide")
     
# Title banner
html_temp = """
    <div style="background-color:teal;padding:13px">
        <h1 style="color:white;text-align:center;">Byte x Brains 💻🧠</h1>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation Pane")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Hypothesis Testing", "Policy Recommendations"])

# File uploader
st.sidebar.markdown("### Upload Your SSI Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("cleaned_ssi_data.csv")  # Default fallback
    return df

df = load_data(uploaded_file)

if df is not None:
    # -----------------------------
    # Dashboard Overview
    # -----------------------------
    if page == "Dashboard Overview":
        st.title("Surgical Site Infection (SSI) Analytics App")
        st.subheader("Key Metrics and Visualizations")

        st.markdown("### Summary Statistics")
        st.write(df.describe())

        # Heatmap
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Infection ratio by county
        st.markdown("### Average Top 10 Counties by Infections Reported")
        top_counties = df.groupby("County")["Infections_Reported"].mean().sort_values(ascending=True).head(10)
        st.bar_chart(top_counties)

        # SIR by Operative Procedure
        st.markdown("### Average SIR by Operative Procedure")
        avg_sir_op = df.groupby("Operative_Procedure")["SIR"].mean().sort_values(ascending=True).dropna()
        st.bar_chart(avg_sir_op)

    # -----------------------------
    # Hypothesis Testing
    # -----------------------------
    elif page == "Hypothesis Testing":
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
            st.info("There is no statistically significant difference in infection ratoio between large and small-sized hospital beds.")

        # Boxplot
        st.markdown("### SIR Distribution by Hospital Size")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="Hospital_Category_RiskAdjustment", y="SIR", data=df)
        plt.xticks(rotation=30)
        st.pyplot(fig2)

    # -----------------------------
    # Policy Recommendations
    # -----------------------------
    elif page == "Policy Recommendations":
        st.title("Health Policy Recommendations")

        user_context = st.text_area("Provide additional context (optional):", "")

        if st.button("Generate Recommendations"):   
            def configure_gemini():
                 """Configure Gemini API with API key from Streamlit secrets"""
                 try:
                      # Get API key from Streamlit secrets
                      api_key = st.secrets["GEMINI_API_KEY"]
                      genai.configure(api_key=api_key)
                      return True
                 except KeyError:
                      st.error("🔑 API key not found in secrets.toml file.")
                      return False
                 except Exception as e:
                      st.error(f"Error configuring API: {str(e)}")
                      return False
             
           
            prompt = (
                "You are an exoerienced health policy analyst, generate 5 actionable and evidence-based policy recommendations "
                "to help reduce the Standardized Surgical Infection Ratio (SIR) across health centers in California."
                "Make the recommendation as simple and relatable as possible."
                f"Use the following context if useful:\n{user_context}"
            )

            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)

            st.markdown("### Tailored Recommendations")
            st.write(response.text)
else:
    st.warning("❗ Please upload a valid CSV file or use the default dataset.")

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import google.generativeai as genai

# Page configuration
st.set_page_config(page_title="SSI Analytics App", page_icon="⚕️", layout="wide")

# Title banner
html_temp = """
    <div style="background-color:teal;padding:13px">
        <h2 style="color:white;text-align:center;">Byte x Brains 💻🧠</h2>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation Pane")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Hypothesis Testing", "Policy Recommendations"])

# File uploader
st.sidebar.markdown("### Upload Your SSI Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Load Data
@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = pd.read_csv("cleaned_ssi_data.csv")  # Default fallback
    return df

df = load_data(uploaded_file)

# Check column structure
expected_cols = ["County", "Infections_Reported", "SIR", "Operative_Procedure", "Hospital_Category_RiskAdjustment"]
if not all(col in df.columns for col in expected_cols):
    st.error("🚫 Uploaded file does not have the expected structure. Please upload a valid SSI dataset.")
    st.stop()

if df is not None:
    if page == "Dashboard Overview":
        st.title("Surgical Site Infection (SSI) Analytics App")
        
        # ABOUT SECTION
        st.markdown("### 🩺 About the App")
        st.markdown("""
        This interactive dashboard is designed to help users analyze and interpret **Surgical Site Infection (SSI)** data 
        from healthcare centers across California. It provides insights into infection trends, hospital risk factors, 
        and performance indicators to support data-driven health policy decisions.

        **How to Navigate:**
        - Use the **sidebar** to switch between different pages.
        - 📊 **Dashboard Overview**: View metrics, trends, and infection rates across counties and procedures.
        - 🧪 **Hypothesis Testing**: Compare infection ratios using statistical methods.
        - 📋 **Policy Recommendations**: Generate evidence-based strategies using AI to improve health outcomes.

        Upload your custom CSV data or explore the built-in dataset to get started!
        """)

        st.subheader("Key Metrics and Visualizations")

        st.markdown("### Summary Statistics")
        st.write(df.describe())

        # Heatmap
        st.markdown("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

        # Infection ratio by county
        st.markdown("### Top 10 Counties by Infections Reported")
        top_counties = df.groupby("County")["Infections_Reported"].mean().sort_values(ascending=False).head(10)
        st.bar_chart(top_counties)

        # SIR by Operative Procedure
        st.markdown("### Average SIR by Operative Procedure")
        avg_sir_op = df.groupby("Operative_Procedure")["SIR"].mean().sort_values(ascending=False).dropna()
        st.bar_chart(avg_sir_op)

    elif page == "Hypothesis Testing":
        st.title("Hypothesis Testing")

        st.markdown("### Comparing SIR between Hospitals by Bed Size")
        small_hospitals = df[df['Hospital_Category_RiskAdjustment'] == 'Smaller hospitals (<250 beds)']['SIR'].dropna()
        large_hospitals = df[df['Hospital_Category_RiskAdjustment'] == 'Larger hospitals (>=250 beds)']['SIR'].dropna()

        t_stat, p_val = stats.ttest_ind(large_hospitals, small_hospitals, equal_var=False)

        st.write("T-statistic: {}".format(t_stat))
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
        plt.tight_layout()
        st.pyplot(fig2)

    elif page == "Policy Recommendations":
        st.title("Health Policy Recommendations")
    
        user_context = st.text_area("Provide additional context (optional):", "")
    
        if st.button("Generate Recommendations"):
            def configure_gemini():
                """Configure Gemini API with API key from Streamlit secrets"""
                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                    genai.configure(api_key=api_key)
                    return True
                except KeyError:
                    st.error("🔑 API key not found in secrets.toml file.")
                    return False
                except Exception as e:
                    st.error(f"Error configuring Gemini API: {str(e)}")
                    return False
    
            if configure_gemini():
                # Extract key findings from the dataset
                avg_sir = df['SIR'].mean()
                top_counties = df.groupby("County")["Infections_Reported"].mean().sort_values(ascending=False).head(3)
                high_sir_procedures = df.groupby("Operative_Procedure")["SIR"].mean().sort_values(ascending=False).dropna().head(3)
    
                top_counties_list = ", ".join(top_counties.index)
                top_procedures_list = ", ".join(high_sir_procedures.index)
    
                data_summary = (
                    f"Here are key findings from the SSI dataset:\n"
                    f"- The average Standardized Infection Ratio (SIR) is approximately **{avg_sir:.2f}**.\n"
                    f"- The counties with the highest reported infections include: **{top_counties_list}**.\n"
                    f"- Procedures associated with the highest infection ratios include: **{top_procedures_list}**.\n"
                )
    
                prompt = (
                    "You are an experienced health policy analyst. Based on the summarized data insights below, "
                    "generate 5 actionable and evidence-based policy recommendations to help reduce the Standardized Surgical Infection Ratio (SIR) across health centers in California. "
                    "Ensure the recommendations are simple, clear, and directly tied to the data.\n\n"
                    f"{data_summary}\n"
                    f"Additional context (if any):\n{user_context}"
                )
    
                model = genai.GenerativeModel("gemini-pro")
                response = model.generate_content(prompt)
    
                st.markdown("### Tailored Recommendations")
                st.write(response.text)

    
else:
    st.warning("❗ Please upload a valid CSV file or use the default dataset.")

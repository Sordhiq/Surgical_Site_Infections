import streamlit as st
import pandas as pd
import numpy as np

# --------------------------------------
# Load Dataset
# --------------------------------------
st.title("🧼 Surgical Site Infections Dashboard – California")

uploaded_file = st.file_uploader("📁 Upload SSI dataset (CSV)", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")

    # --------------------------------------
    # Data Cleaning Functions
    # --------------------------------------
    def clean_data(data):
        # Compute SIR
        valid_pred = (data['Infections_Predicted'] >= 0.2)
        valid_calc = valid_pred & (data['Infections_Predicted'] != 0) & (data['SIR'].isnull())
        data.loc[valid_calc, 'SIR'] = data.loc[valid_calc, 'Infections_Reported'] / data.loc[valid_calc, 'Infections_Predicted']
        data['SIR'] = data.groupby(['HAI', 'Operative_Procedure'])['SIR'].transform(lambda x: x.fillna(x.median()))

        # Impute CIs
        for col in ['SIR_CI_95_Lower_Limit', 'SIR_CI_95_Upper_Limit']:
            data[col] = data.groupby(['HAI', 'Operative_Procedure'])[col].transform(lambda x: x.fillna(x.median()))

        # Define Comparison
        def determine_comparison(row):
            if pd.isnull(row['SIR_CI_95_Lower_Limit']) or pd.isnull(row['SIR_CI_95_Upper_Limit']):
                return np.nan
            elif row['SIR_CI_95_Lower_Limit'] > 1:
                return "Worse than National"
            elif row['SIR_CI_95_Upper_Limit'] < 1:
                return "Better than National"
            else:
                return "No Different"
        data['Comparison'] = data['Comparison'].fillna(data.apply(determine_comparison, axis=1))

        # Met 2020 Goal
        def met_2020_goal(row):
            if pd.isna(row['SIR']) or row['Year'] < 2021:
                return np.nan
            return "Yes" if row['SIR'] < 0.70 else "No"
        data['Met_2020_Goal'] = data['Met_2020_Goal'].fillna(data.apply(met_2020_goal, axis=1))

        # SIR_2015 Imputation
        valid_sir2015 = (data['Infections_Predicted'] >= 0.2) & (data['SIR_2015'].isnull())
        data.loc[valid_sir2015, 'SIR_2015'] = data.groupby(['Facility_ID', 'HAI'])['SIR_2015'].transform(lambda x: x.fillna(x.median()))
        data['SIR_2015'] = data.groupby('HAI')['SIR_2015'].transform(lambda x: x.fillna(x.median()))
        data['SIR_2015'].fillna(data['SIR_2015'].median(), inplace=True)

        # Missing Reason and Flag
        def sir_missing_reason(row):
            if not pd.isna(row['SIR']):
                return "Calculated"
            elif row['Infections_Predicted'] < 0.2:
                return "Below threshold (<0.2)"
            elif row['Infections_Predicted'] == 0:
                return "Zero predicted"
            else:
                return "Unknown"
        data['SIR_Missing_Reason'] = data.apply(sir_missing_reason, axis=1)
        data['SIR_missing_flag'] = data['SIR'].isnull().astype(int)

        return data

    # Clean the data
    data_cleaned = clean_data(data)

    # --------------------------------------
    # Data Overview
    # --------------------------------------
    st.subheader("📊 Dataset Overview")
    st.write(data_cleaned.head())

    st.subheader("🩺 Missing Value Summary")
    st.write(data_cleaned.isnull().sum()[data_cleaned.isnull().sum() > 0])

    # --------------------------------------
    # Visualizations
    # --------------------------------------
    st.subheader("📈 SIR Distribution by Year")
    st.bar_chart(data_cleaned.groupby('Year')['SIR'].median())

    st.subheader("🏥 SIR by Facility Type")
    if 'Facility_Type' in data_cleaned.columns:
        st.bar_chart(data_cleaned.groupby('Facility_Type')['SIR'].median())

    st.subheader("✅ Met 2020 Goal by Year")
    met_goal_counts = data_cleaned.groupby(['Year', 'Met_2020_Goal']).size().unstack()
    st.bar_chart(met_goal_counts)

    # --------------------------------------
    # Download Cleaned Data
    # --------------------------------------
    st.subheader("⬇️ Download Cleaned Dataset")
    csv = data_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "cleaned_ssi_data.csv", "text/csv")

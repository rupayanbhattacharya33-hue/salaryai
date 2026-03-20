import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SalaryAI", page_icon="💰", layout="wide")

BASE = "backend/models/"
model      = pickle.load(open(BASE + "salary_model.pkl", "rb"))
le_job     = pickle.load(open(BASE + "le_job.pkl", "rb"))
le_edu     = pickle.load(open(BASE + "le_edu.pkl", "rb"))
le_country = pickle.load(open(BASE + "le_country.pkl", "rb"))
le_gender  = pickle.load(open(BASE + "le_gender.pkl", "rb"))
meta       = json.load(open(BASE + "meta.json"))

INR_RATE = 83.5

EXTRA_COUNTRIES = {
    'India':    0.35,
    'Pakistan': 0.22,
    'UAE':      0.75,
}

st.markdown("""
<style>
.big-salary { font-size: 2.8rem; font-weight: 700; color: #6366f1; }
.inr-salary { font-size: 1.4rem; color: #22c55e; font-weight: 600; }
.label      { font-size: 0.85rem; color: #888; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("💰 SalaryAI — Job Salary Predictor")
st.caption("Predict your expected salary based on role, experience, education and location · Powered by Random Forest ML (R² = 0.965)")

st.divider()

col1, col2 = st.columns([1, 1.6])

with col1:
    st.subheader("Enter Your Details")
    job_title   = st.selectbox("Job Title", sorted(meta['job_titles']))
    experience  = st.slider("Years of Experience", 0, 30, 3)
    age         = st.slider("Age", 18, 65, 28)
    education   = st.selectbox("Education Level", meta['education'])
    all_countries = sorted(meta['countries'] + ['India', 'Pakistan', 'UAE'])
    country     = st.selectbox("Country", all_countries)
    gender      = st.selectbox("Gender", meta['genders'])
    predict_btn = st.button("🔍 Predict My Salary", use_container_width=True, type="primary")

with col2:
    if predict_btn:
        is_extra     = country in EXTRA_COUNTRIES
        base_country = 'USA' if is_extra else country
        adjustment   = EXTRA_COUNTRIES[country] if is_extra else 1.0

        input_df = pd.DataFrame([{
            'Age':        age,
            'Gender':     le_gender.transform([gender])[0],
            'Education':  le_edu.transform([education])[0],
            'Job Title':  le_job.transform([job_title])[0],
            'Experience': experience,
            'Country':    le_country.transform([base_country])[0],
        }])

        salary_usd = model.predict(input_df)[0] * adjustment
        salary_inr = salary_usd * INR_RATE
        salary_min = salary_usd * 0.85
        salary_max = salary_usd * 1.15

        st.subheader("Prediction Result")

        if is_extra:
            st.caption(f"⚠️ {country} estimate is based on regional cost-of-living adjustment from USA baseline")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown('<p class="label">Predicted Salary (USD)</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="big-salary">${salary_usd:,.0f}</p>', unsafe_allow_html=True)
        with r2:
            st.markdown('<p class="label">Equivalent in INR (₹83.5/$)</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="inr-salary">₹{salary_inr:,.0f}</p>', unsafe_allow_html=True)

        st.caption(f"Estimated range: ${salary_min:,.0f} — ${salary_max:,.0f} USD")
        st.divider()

        m1, m2, m3 = st.columns(3)
        m1.metric("Job Title",  job_title)
        m2.metric("Experience", f"{experience} years")
        m3.metric("Country",    country)

        st.divider()

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=salary_usd,
            number={'prefix': "$", 'valueformat': ",.0f"},
            title={'text': "Predicted Salary (USD)"},
            gauge={
                'axis': {'range': [0, 260000]},
                'bar': {'color': "#6366f1"},
                'steps': [
                    {'range': [0, 70000],     'color': "#fee2e2"},
                    {'range': [70000, 130000], 'color': "#fef9c3"},
                    {'range': [130000, 260000],'color': "#dcfce7"},
                ],
                'threshold': {'line': {'color': "#6366f1", 'width': 3}, 'value': salary_usd}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.info("Fill in your details on the left and click **Predict My Salary**")
        st.image("https://img.icons8.com/clouds/200/money-bag.png", width=150)

st.divider()
st.subheader("📊 Dataset Insights")

df = pd.read_csv("notebook/Salary_Data_Based_country_and_race.csv")
df = df.drop('Unnamed: 0', axis=1)
df.columns = ['Age', 'Gender', 'Education', 'Job Title', 'Experience', 'Salary', 'Country', 'Race']
edu_map = {
    "Bachelor's": "Bachelor's", "Bachelor's Degree": "Bachelor's",
    "Master's": "Master's",     "Master's Degree": "Master's",
    'PhD': 'PhD', 'phD': 'PhD', 'High School': 'High School'
}
df['Education'] = df['Education'].map(edu_map)
df = df.dropna()

t1, t2, t3, t4 = st.tabs(["By Job Title", "By Country", "By Education", "By Experience"])

with t1:
    top_n = st.slider("Show top N jobs", 5, 30, 15, key="topn")
    top_jobs = df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(top_n).reset_index()
    fig = px.bar(top_jobs, x='Salary', y='Job Title', orientation='h',
                 color='Salary', color_continuous_scale='Viridis',
                 title=f"Top {top_n} Highest Paying Jobs")
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with t2:
    country_df = df.groupby('Country')['Salary'].mean().reset_index()
    fig2 = px.bar(country_df, x='Country', y='Salary', color='Salary',
                  color_continuous_scale='Blues', title="Average Salary by Country")
    st.plotly_chart(fig2, use_container_width=True)

with t3:
    edu_order = ['High School', "Bachelor's", "Master's", 'PhD']
    edu_df = df.groupby('Education')['Salary'].mean().reindex(edu_order).reset_index()
    fig3 = px.bar(edu_df, x='Education', y='Salary', color='Salary',
                  color_continuous_scale='Greens', title="Average Salary by Education Level")
    st.plotly_chart(fig3, use_container_width=True)

with t4:
    sample_size = min(1000, len(df))
    fig4 = px.scatter(df.sample(sample_size, random_state=42), x='Experience', y='Salary',
                      color='Country', opacity=0.6, title="Salary vs Experience")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption("SalaryAI · Built with Python, scikit-learn, Streamlit · INR rate: ₹83.5 per $1 USD")
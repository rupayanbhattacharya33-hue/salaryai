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
EXTRA_COUNTRIES = {'India': 0.35, 'Pakistan': 0.22, 'UAE': 0.75}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}
.hero h1 { font-size: 2.8rem; font-weight: 700; margin: 0; }
.hero p  { font-size: 1.1rem; opacity: 0.85; margin-top: 0.5rem; }

.result-card {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    padding: 1.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}
.result-card .amount { font-size: 3rem; font-weight: 700; }
.result-card .label  { font-size: 0.9rem; opacity: 0.85; }

.inr-card {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    padding: 1.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}
.inr-card .amount { font-size: 2rem; font-weight: 700; }
.inr-card .label  { font-size: 0.9rem; opacity: 0.85; }

.stat-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stat-card .val { font-size: 1.4rem; font-weight: 700; color: #6366f1; }
.stat-card .lbl { font-size: 0.8rem; color: #6b7280; margin-top: 2px; }

.compare-card {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
.tip-box {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #92400e;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>💰 SalaryAI</h1>
    <p>Predict your market salary instantly · Powered by Random Forest ML · R² = 0.965</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 🧑‍💼 Your Profile")

    job_title   = st.selectbox("Job Title", sorted(meta['job_titles']))
    experience  = st.slider("Years of Experience", 0, 30, 3)
    age         = st.slider("Age", 18, 65, 28)
    education   = st.selectbox("Education Level", meta['education'])
    all_countries = sorted(meta['countries'] + ['India', 'Pakistan', 'UAE'])
    country     = st.selectbox("Country", all_countries)
    gender      = st.selectbox("Gender", meta['genders'])

    st.markdown("### 🔁 Compare With")
    compare_exp = st.slider("Compare experience level", 0, 30, experience + 5)
    compare_country = st.selectbox("Compare country", all_countries, index=all_countries.index('USA') if 'USA' in all_countries else 0)

    predict_btn = st.button("🔍 Predict My Salary", use_container_width=True, type="primary")

with col2:
    if predict_btn:
        def get_salary(exp, ctry):
            is_extra     = ctry in EXTRA_COUNTRIES
            base_country = 'USA' if is_extra else ctry
            adjustment   = EXTRA_COUNTRIES[ctry] if is_extra else 1.0
            inp = pd.DataFrame([{
                'Age': age,
                'Gender': le_gender.transform([gender])[0],
                'Education': le_edu.transform([education])[0],
                'Job Title': le_job.transform([job_title])[0],
                'Experience': exp,
                'Country': le_country.transform([base_country])[0],
            }])
            return model.predict(inp)[0] * adjustment

        salary_usd   = get_salary(experience, country)
        compare_usd  = get_salary(compare_exp, compare_country)
        salary_inr   = salary_usd * INR_RATE
        salary_min   = salary_usd * 0.85
        salary_max   = salary_usd * 1.15
        diff_pct     = ((compare_usd - salary_usd) / salary_usd) * 100

        if country in EXTRA_COUNTRIES:
            st.caption(f"⚠️ {country} uses regional cost-of-living adjustment from USA baseline")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="result-card">
                <div class="label">Your Predicted Salary</div>
                <div class="amount">${salary_usd:,.0f}</div>
                <div class="label">USD per year</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="inr-card">
                <div class="label">Equivalent in INR</div>
                <div class="amount">₹{salary_inr:,.0f}</div>
                <div class="label">@ ₹83.5 per $1</div>
            </div>""", unsafe_allow_html=True)

        st.caption(f"Estimated range: ${salary_min:,.0f} — ${salary_max:,.0f} USD")
        st.divider()

        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(f'<div class="stat-card"><div class="val">{experience}y</div><div class="lbl">Experience</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="stat-card"><div class="val">{country}</div><div class="lbl">Country</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="stat-card"><div class="val">{education}</div><div class="lbl">Education</div></div>', unsafe_allow_html=True)
        s4.markdown(f'<div class="stat-card"><div class="val">${salary_usd/12:,.0f}</div><div class="lbl">Per Month</div></div>', unsafe_allow_html=True)

        st.divider()

        arrow = "📈" if compare_usd > salary_usd else "📉"
        st.markdown(f"""
        <div class="compare-card">
            <strong>{arrow} Comparison: {compare_exp} yrs exp in {compare_country}</strong><br>
            Predicted: <strong>${compare_usd:,.0f}</strong> — 
            that's <strong>{abs(diff_pct):.1f}% {"more" if compare_usd > salary_usd else "less"}</strong> than your profile
        </div>""", unsafe_allow_html=True)

        monthly = salary_usd / 12
        if monthly < 3000:
            tip = "💡 Tip: Upskilling to a Senior role could increase your salary by 40-60%."
        elif monthly < 6000:
            tip = "💡 Tip: Adding cloud certifications (AWS/GCP) typically adds $10-20K to your package."
        else:
            tip = "💡 Tip: You're in the top salary bracket! Leadership roles could push you further."

        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

        st.divider()

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=salary_usd,
            delta={'reference': compare_usd, 'valueformat': ",.0f"},
            number={'prefix': "$", 'valueformat': ",.0f"},
            title={'text': f"Your Salary vs {compare_country} ({compare_exp}y)"},
            gauge={
                'axis': {'range': [0, 260000]},
                'bar': {'color': "#6366f1"},
                'steps': [
                    {'range': [0, 70000],     'color': "#fee2e2"},
                    {'range': [70000, 130000], 'color': "#fef9c3"},
                    {'range': [130000, 260000],'color': "#dcfce7"},
                ],
                'threshold': {'line': {'color': "#f5576c", 'width': 3}, 'value': compare_usd}
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=50, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #9ca3af;">
            <div style="font-size:4rem">💼</div>
            <div style="font-size:1.1rem; margin-top:1rem">Fill in your profile and click Predict</div>
            <div style="font-size:0.85rem; margin-top:0.5rem">Get your salary + INR equivalent + smart comparison</div>
        </div>""", unsafe_allow_html=True)

st.divider()
st.markdown("### 📊 Market Insights")

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

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Records",    f"{len(df):,}")
m2.metric("Avg Salary",       f"${df['Salary'].mean():,.0f}")
m3.metric("Highest Salary",   f"${df['Salary'].max():,.0f}")
m4.metric("Job Titles",       f"{df['Job Title'].nunique()}")

t1, t2, t3, t4, t5 = st.tabs(["🏆 By Job Title", "🌍 By Country", "🎓 By Education", "📈 By Experience", "⚖️ Gender Gap"])

with t1:
    top_n = st.slider("Show top N jobs", 5, 30, 15, key="topn")
    top_jobs = df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(top_n).reset_index()
    fig = px.bar(top_jobs, x='Salary', y='Job Title', orientation='h',
                 color='Salary', color_continuous_scale='Viridis',
                 title=f"Top {top_n} Highest Paying Jobs")
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'},
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with t2:
    country_df = df.groupby('Country')['Salary'].mean().reset_index()
    fig2 = px.bar(country_df, x='Country', y='Salary', color='Salary',
                  color_continuous_scale='Blues', title="Average Salary by Country",
                  text_auto='.2s')
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

with t3:
    edu_order = ['High School', "Bachelor's", "Master's", 'PhD']
    edu_df = df.groupby('Education')['Salary'].mean().reindex(edu_order).reset_index()
    fig3 = px.bar(edu_df, x='Education', y='Salary', color='Salary',
                  color_continuous_scale='Greens', title="Average Salary by Education Level",
                  text_auto='.2s')
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

with t4:
    sample_size = min(1000, len(df))
    fig4 = px.scatter(df.sample(sample_size, random_state=42), x='Experience', y='Salary',
                      color='Country', opacity=0.6, title="Salary vs Experience",
                      trendline="lowess")
    fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4, use_container_width=True)

with t5:
    gender_df = df.groupby(['Gender', 'Education'])['Salary'].mean().reset_index()
    fig5 = px.bar(gender_df, x='Education', y='Salary', color='Gender',
                  barmode='group', title="Salary by Gender and Education",
                  color_discrete_map={'Male': '#6366f1', 'Female': '#f43f5e'},
                  category_orders={'Education': edu_order})
    fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.caption("SalaryAI · Random Forest ML · R²=0.965 · Built with Python, scikit-learn, Streamlit, Plotly · INR rate ₹83.5/$1")
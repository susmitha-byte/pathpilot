import streamlit as st
import ollama
import vector_db


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="AI Career Intelligence", layout="wide")


# -------------------------------
# Premium Dark UI
# -------------------------------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #1c1d1f;
    color: #ff9f43;
}
/* Make all input labels white */
label {
    color: #ffffff !important;
}

/* Extra safety for Streamlit form labels */
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label {
    color: #ffffff !important;
}


.card {
    background: #2d2f31;
    border: 1px solid #3e4143;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 20px;
}

.main-title {
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(90deg, #CEC0FF, #A435F0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-header {
    font-size: 20px;
    font-weight: 600;
    border-left: 5px solid #A435F0;
    padding-left: 10px;
    margin-bottom: 15px;
}

.stButton>button {
    background-color: #A435F0;
    color: white !important;
    border-radius: 4px;
    font-weight: 700;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------
# Ollama Function
# -------------------------------
def chat_with_gemma(sys_msg, user_msg):
    try:
        res = ollama.chat(
            model="gemma3:latest",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg}
            ]
        )
        return res["message"]["content"]
    except Exception as e:
        return f"Model Error: {str(e)}"


# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("## 🚀 CAREER AI")
    app_mode = st.radio(
        "Navigation",
        ["🎯 Career Advisor", "📈 Growth Planner", "📄 Resume Optimizer"]
    )


# ===============================
# TAB 1: Career Advisor
# ===============================
if app_mode == "🎯 Career Advisor":

    st.markdown('<p class="main-title">AI Career Intelligence</p>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">Candidate Profile</p>', unsafe_allow_html=True)

        cgpa = st.number_input("Academic CGPA", 0.0, 10.0, 0.0)
        interns = st.number_input("Completed Internships", 0, 20, 0)
        skills = st.text_area("Core Skills")
        role = st.text_input("Desired Role")

        analyze_btn = st.button("RUN PREDICTIVE ANALYSIS")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze_btn:
        if cgpa == 0 or interns == 0 or not skills.strip() or not role.strip():
            st.error("⚠️ Please fill all fields.")
        else:
            with st.spinner("Analyzing profile..."):

                prob = vector_db.predict_placement(cgpa, interns)
                salary = vector_db.predict_salary(cgpa, interns)

                with right:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.metric("Placement Probability", f"{int(prob*100)}%")
                    st.metric("Expected Salary", f"${int(salary):,}")
                    st.progress(prob)

                    sys_msg = "Explain placement prediction based on academic profile."
                    user_msg = f"CGPA: {cgpa}, Internships: {interns}, Skills: {skills}, Role: {role}"

                    advice = chat_with_gemma(sys_msg, user_msg)
                    st.markdown(f"### Coach Insight\n{advice}")

                    st.markdown('</div>', unsafe_allow_html=True)

                st.session_state.last_p = {
                    "cgpa": cgpa,
                    "interns": interns,
                    "skills": skills,
                    "role": role
                }


# ===============================
# TAB 2: Growth Planner
# ===============================
elif app_mode == "📈 Growth Planner":

    st.markdown('<p class="main-title">Personalized Growth Roadmap</p>', unsafe_allow_html=True)

    if "last_p" not in st.session_state:
        st.info("Complete Career Advisor first.")
    else:
        p = st.session_state.last_p

        if st.button("GENERATE 6-MONTH STRATEGIC PLAN"):
            with st.spinner("Generating roadmap..."):
                sys_msg = "Create a structured 6-month career roadmap."
                user_msg = f"Role: {p['role']}, Skills: {p['skills']}"
                roadmap = chat_with_gemma(sys_msg, user_msg)
                st.markdown(roadmap)


# ===============================
# TAB 3: Resume Optimizer
# ===============================
elif app_mode == "📄 Resume Optimizer":

    st.markdown('<p class="main-title">AI Resume Optimizer</p>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        target_role = st.text_input("Target Job Role")
    with col2:
        years_exp = st.text_input("Years of Experience")

    resume_text = st.text_area("Paste Your Resume", height=250)

    analyze_resume_btn = st.button("Analyze Resume")

    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_resume_btn:
        if not resume_text.strip():
            st.error("Please paste resume content.")
        elif not target_role.strip():
            st.error("Please enter target role.")
        else:
            with st.spinner("Analyzing resume..."):

                search_query = f"High salary profiles for {target_role}"
                retrieved_context = vector_db.get_retriever(search_query)
                context_str = "\n".join(retrieved_context)

                sys_prompt = (
                    "You are an ATS expert. Analyze resume using dataset context. "
                    "Provide structured response with score, strengths, weaknesses, "
                    "missing skills, optimization suggestions."
                )

                user_prompt = f"""
Resume:
{resume_text}

Target Role:
{target_role} ({years_exp})

Dataset Context:
{context_str}
"""

                result = chat_with_gemma(sys_prompt, user_prompt)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### ATS Analysis Result")
                st.markdown(result)
                st.markdown('</div>', unsafe_allow_html=True)

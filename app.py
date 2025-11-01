import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---- LOAD DATA ----
with open("Untitled (2).json", "r", encoding="utf-8") as f:
    careers = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI Career Roadmap Generator", page_icon="🚀", layout="wide")

st.title("🚀 AI Career Roadmap Generator")
st.markdown("Enter your **skills** and **interests** to discover your ideal career path and personalized roadmap.")

# ---- USER INPUT ----
user_skills_input = st.text_input("🧠 Enter your skills (comma-separated):", "Python, SQL, pandas")
user_interests_input = st.text_input("💡 Enter your interests (comma-separated):", "AI, analytics, data science")

if st.button("🎯 Generate Career Matches"):
    user_skills = [s.strip().lower() for s in user_skills_input.split(",") if s.strip()]
    user_interests = [i.strip().lower() for i in user_interests_input.split(",") if i.strip()]

    career_texts = [" ".join(c["required_skills"] + c["interest_tags"]) for c in careers]

    vectorizer = TfidfVectorizer().fit(career_texts)
    career_vectors = vectorizer.transform(career_texts)
    user_vector = vectorizer.transform([" ".join(user_skills + user_interests)])
    similarities = cosine_similarity(user_vector, career_vectors)[0]

    top_indices = np.argsort(similarities)[::-1][:3]
    recommendations = [careers[i] for i in top_indices]
    st.session_state["recommendations"] = recommendations

# ---- SHOW RECOMMENDATIONS ----
if "recommendations" in st.session_state:
    st.subheader("🎯 Top Career Matches For You")
    rec_cols = st.columns(3)
    for idx, c in enumerate(st.session_state["recommendations"]):
        with rec_cols[idx]:
            st.markdown(f"### 🧭 {c['career']}")
            st.caption(f"**Key Skills:** {', '.join(c['required_skills'][:5])}")
            st.caption(f"**Focus Areas:** {', '.join(c['interest_tags'][:3])}")

    selected_career_name = st.selectbox(
        "Select your preferred career:",
        [c["career"] for c in st.session_state["recommendations"]],
    )
    selected_career_data = next(c for c in st.session_state["recommendations"] if c["career"] == selected_career_name)

    # Level detection
    user_skills = [s.strip().lower() for s in user_skills_input.split(",") if s.strip()]
    career_skills = [s.lower() for s in selected_career_data["required_skills"]]
    skill_match = len(set(user_skills) & set(career_skills)) / max(1, len(career_skills))

    if skill_match < 0.3:
        default_level = "Beginner"
    elif skill_match < 0.7:
        default_level = "Intermediate"
    else:
        default_level = "Advanced"

    selected_level = st.radio(
        "Select your current level:",
        ["Beginner", "Intermediate", "Advanced"],
        index=["Beginner", "Intermediate", "Advanced"].index(default_level)
    )

    # ✅ NEW: Button to confirm before showing roadmap
    show_roadmap = st.button("🗺️ Generate My Roadmap")

    if show_roadmap:
        st.markdown("---")
        st.subheader(f"🗺️ Personalized Roadmap for {selected_career_name} ({selected_level})")

        roadmap_steps = selected_career_data["roadmap"][selected_level]
        cols = st.columns(2)
        for i, step in enumerate(roadmap_steps):
            with cols[i % 2]:
                st.markdown(f"""
                <div style='background-color:#1E1E1E;padding:15px;border-radius:10px;margin-bottom:10px;
                            border-left:5px solid #00BFFF'>
                    <h4 style='color:#00BFFF;'>Step {i+1}</h4>
                    <p style='color:#ddd;'>{step}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 📚 Recommended Resources:")
        for r in selected_career_data["resources"]:
            st.markdown(f"🔗 {r}")

        # ✅ Save all data for the tracker
        st.session_state["selected_career"] = selected_career_name
        st.session_state["selected_level"] = selected_level
        st.session_state["selected_tasks"] = roadmap_steps

        # ✅ Confirmation message
        st.success("✅ Career and roadmap saved successfully!")

        # ✅ Navigation button to tracker
        from streamlit_extras.switch_page_button import switch_page

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("You can now track your progress step-by-step below 👇")

        # Center the button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Go to Learning Progress Tracker"):
                switch_page("learning_progress_tracker")

    
    
    
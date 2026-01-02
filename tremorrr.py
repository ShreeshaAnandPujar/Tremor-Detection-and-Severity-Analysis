import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from pathlib import Path

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Hand Tremor Detection", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>🧠 Hand Tremor Detection & Severity Analyzer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Educational & Screening Prototype (Not Diagnostic)</p>",
    unsafe_allow_html=True
)

# =========================================================
# SESSION STATE
# =========================================================
if "started" not in st.session_state:
    st.session_state.started = False
if "tremor_history" not in st.session_state:
    st.session_state.tremor_history = []

# =========================================================
# PATIENT DETAILS FORM (SHOWN ONLY ONCE)
# =========================================================
if not st.session_state.started:
    st.subheader("👤 Patient Information")

    with st.form("patient_form"):
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=1, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        start_btn = st.form_submit_button("▶ Start Tremor Analysis")

    if start_btn:
        st.session_state.started = True
        st.session_state.name = name
        st.session_state.age = age
        st.session_state.gender = gender
        st.rerun()

    st.stop()

# =========================================================
# MEDIAPIPE INITIALIZATION
# =========================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def tremor_severity(score):
    if score < 0.002:
        return "NORMAL", "#00c853"
    elif score < 0.006:
        return "MILD", "#ffab00"
    else:
        return "SEVERE", "#d50000"

# =========================================================
# UI LAYOUT
# =========================================================
col_cam, col_data = st.columns([2.5, 1.5])

frame_view = col_cam.image([])
graph_view = col_data.empty()
status_view = col_data.empty()

st.markdown(
    f"""
    <h3 style="text-align:center;">
        Patient: {st.session_state.name} |
        Age: {st.session_state.age} |
        Gender: {st.session_state.gender}
    </h3>
    """,
    unsafe_allow_html=True
)

# =========================================================
# CAMERA + ANALYSIS LOOP
# =========================================================
cap = cv2.VideoCapture(0)
prev_positions = []
MAX_FRAMES = 30

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Camera not accessible. Check permissions.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    tremor_score = 0.0

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark[8]  # Index fingertip
            pos = (lm.x, lm.y)

            prev_positions.append(pos)
            if len(prev_positions) > MAX_FRAMES:
                prev_positions.pop(0)

            if len(prev_positions) >= 2:
                movements = [
                    math.dist(prev_positions[i], prev_positions[i - 1])
                    for i in range(1, len(prev_positions))
                ]
                tremor_score = np.var(movements)
                st.session_state.tremor_history.append(tremor_score)

                if len(st.session_state.tremor_history) > 200:
                    st.session_state.tremor_history.pop(0)

    # ---------------- DISPLAY CAMERA ----------------
    frame_view.image(frame, channels="BGR")

    # ---------------- STATUS DISPLAY ----------------
    severity, color = tremor_severity(tremor_score)

    status_view.markdown(
        f"""
        <div style="text-align:center;">
            <h1 style="color:{color};">{severity}</h1>
            <h2>Tremor Score</h2>
            <h1 style="color:{color};">{tremor_score:.5f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------- LIVE GRAPH ----------------
    fig, ax = plt.subplots()
    ax.plot(st.session_state.tremor_history, linewidth=2)
    ax.set_title("Tremor Intensity Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Tremor Score")
    ax.grid(True)
    graph_view.pyplot(fig)
    plt.close(fig)

    time.sleep(0.03)

cap.release()

# =========================================================
# PDF REPORT GENERATION (SAVED TO DESKTOP)
# =========================================================
st.markdown("---")
st.markdown("## 📄 Generate Tremor Report")

if st.button("Generate PDF Report"):
    desktop = Path.home() / "Desktop"
    graph_img = desktop / "tremor_graph.png"
    pdf_file = desktop / "Tremor_Report.pdf"

    # Save graph image
    plt.figure()
    plt.plot(st.session_state.tremor_history, linewidth=2)
    plt.title("Tremor Intensity Over Time")
    plt.xlabel("Time")
    plt.ylabel("Tremor Score")
    plt.grid(True)
    plt.savefig(graph_img)
    plt.close()

    # Create PDF
    c = canvas.Canvas(str(pdf_file), pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Hand Tremor Screening Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Patient Name: {st.session_state.name}")
    c.drawString(50, height - 130, f"Age: {st.session_state.age}")
    c.drawString(50, height - 160, f"Gender: {st.session_state.gender}")
    c.drawString(50, height - 190, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}")

    avg_score = np.mean(st.session_state.tremor_history)
    sev, _ = tremor_severity(avg_score)

    c.drawString(50, height - 230, f"Average Tremor Score: {avg_score:.5f}")
    c.drawString(50, height - 260, f"Tremor Severity: {sev}")

    c.drawImage(str(graph_img), 50, height - 550, width=400, height=250)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 80, "Disclaimer: This is not a diagnostic medical device.")
    c.drawString(50, 60, "For educational and screening purposes only.")

    c.save()

    st.success("✅ Tremor report saved to Desktop as **Tremor_Report.pdf**")

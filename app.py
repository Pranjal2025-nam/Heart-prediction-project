import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================================
#  PAGE SETUP
# =====================================================================
st.set_page_config(page_title="Heart Health Checker", page_icon="❤️", layout="centered")

THRESHOLD = 0.3  # final threshold (chosen to catch more patients = higher recall)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 820px;}
.hero {background: linear-gradient(135deg, #e63946 0%, #b5179e 100%);
       padding: 2rem 1.5rem; border-radius: 22px; text-align: center; color: #fff;
       box-shadow: 0 10px 30px rgba(230,57,70,.25); margin-bottom: 1.2rem;}
.hero h1 {margin: 0; font-size: 2.2rem; color: #fff;}
.hero p  {margin: .4rem 0 0 0; opacity: .95; font-size: 1.05rem; color: #fff;}
.steps {display: flex; gap: .6rem; margin: .8rem 0 1.2rem 0; flex-wrap: wrap;}
.step  {flex: 1; min-width: 150px; background: rgba(128,128,128,.10); border-radius: 14px;
        padding: .7rem .9rem; text-align: center; font-size: .92rem;}
.step b {display: block; font-size: 1.05rem;}
.explain {background: rgba(66,135,245,.12); border-left: 5px solid #4287f5;
          padding: .7rem 1rem; border-radius: 10px; margin: .4rem 0 1rem 0; font-size: .95rem;}
.explain b {color: #4287f5;}
.badge {display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .85rem;
        font-weight: 600; margin: .2rem 0 .8rem 0;}
.b-good {background: rgba(42,157,92,.18); color: #2a9d5c;}
.b-warn {background: rgba(244,162,38,.20); color: #d98200;}
.b-bad  {background: rgba(230,57,70,.18); color: #e63946;}
.result {padding: 1.6rem 1.2rem; border-radius: 22px; text-align: center; margin: 1rem 0;}
.r-low  {background: rgba(42,157,92,.14);  border: 2px solid #2a9d5c;}
.r-mid  {background: rgba(244,162,38,.16); border: 2px solid #f4a226;}
.r-high {background: rgba(230,57,70,.14);  border: 2px solid #e63946;}
.result .pct {font-size: 3.2rem; font-weight: 800; margin: 0; line-height: 1.1;}
.result h3 {margin: .3rem 0;}
.meter {position: relative; height: 18px; border-radius: 999px; margin: 1.6rem 0 .4rem 0;
        background: linear-gradient(90deg, #2a9d5c 0%, #f4a226 45%, #e63946 100%);}
.pin  {position: absolute; top: -8px; width: 6px; height: 34px; background: #222;
       border: 2px solid #fff; border-radius: 4px; transform: translateX(-50%);}
.tick {position: absolute; top: -4px; width: 2px; height: 26px; background: rgba(255,255,255,.9);}
.meter-labels {display: flex; justify-content: space-between; font-size: .8rem; opacity: .75;}
.factor {display: flex; align-items: center; gap: .6rem; margin: .35rem 0; font-size: .95rem;}
.factor .name {width: 46%;}
.factor .bar-wrap {flex: 1; background: rgba(128,128,128,.15); border-radius: 8px; height: 14px;}
.factor .bar {height: 14px; border-radius: 8px;}
.up   {background: #e63946;}
.down {background: #2a9d5c;}
.tip  {background: rgba(128,128,128,.10); padding: .7rem 1rem; border-radius: 12px; margin: .4rem 0;}
.footer {text-align: center; font-size: .85rem; opacity: .7; margin-top: 1.5rem;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  LOAD MODEL
# =====================================================================
@st.cache_resource
def load_files():
    return (joblib.load("heart_model.pkl"),
            joblib.load("scaler.pkl"),
            joblib.load("columns.pkl"))


model, scaler, columns = load_files()

# =====================================================================
#  OPTIONS + PLAIN-LANGUAGE EXPLANATIONS
# =====================================================================
CP = {
    "No chest pain (Asymptomatic)": ("ASY",
        "You feel <b>no chest pain</b>. This sounds safe, but in the data many heart-disease "
        "patients had <b>no pain at all</b>, so the model treats it as a warning sign."),
    "Pain not from the heart (Non-Anginal)": ("NAP",
        "You feel chest pain, but doctors think it is <b>not caused by the heart</b> "
        "(for example muscle pain or acidity). Usually lower risk."),
    "Slightly heart-like pain (Atypical Angina)": ("ATA",
        "The pain <b>partly looks like</b> heart pain but not fully. Usually lower risk, "
        "but a doctor should still check it."),
    "Classic heart pain (Typical Angina)": ("TA",
        "Pressure or tightness in the chest that comes when the heart <b>does not get enough blood</b>, "
        "often while walking or under stress."),
}
EXANG = {
    "No": "You do <b>not</b> feel chest pain when you exercise, run or climb stairs. Good sign.",
    "Yes": "You <b>feel chest pain when you exercise</b> or climb stairs. This is a strong warning "
           "sign for the heart.",
}
FBS = {
    "No": "Your fasting blood sugar is <b>120 mg/dl or lower</b> (normal).",
    "Yes": "Your fasting blood sugar is <b>above 120 mg/dl</b>. High sugar over time can harm the heart.",
}
ECG = {
    "Normal": ("Normal", "The heart's electrical test at rest looks <b>normal</b>."),
    "ST-T wave abnormality": ("ST", "The ECG shows a <b>small abnormal wave</b>. It can mean the heart "
                                    "is under strain."),
    "Thickened heart wall (LVH)": ("LVH", "The heart's wall is <b>thicker than normal</b> (often linked "
                                          "to long-term high blood pressure)."),
}
SLOPE = {
    "Upsloping (usually healthy)": ("Up", "The ECG line <b>goes up</b> at peak exercise. This is the "
                                          "healthiest pattern."),
    "Flat (more risk)": ("Flat", "The ECG line stays <b>flat</b> at peak exercise. This is linked with "
                                 "more risk."),
    "Downsloping (high risk)": ("Down", "The ECG line <b>goes down</b> at peak exercise. This is the "
                                        "pattern most linked with heart problems."),
}

# default values (also used by the example buttons)
DEFAULTS = dict(age=50, sex="Male", cp=list(CP)[1], exang="No", bp=120, maxhr=150,
                unk_chol=False, chol=200, fbs="No", ecg="Normal",
                oldpeak=1, slope=list(SLOPE)[0], checked=False)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_example(kind):
    if kind == "healthy":
        vals = dict(age=35, sex="Female", cp=list(CP)[2], exang="No", bp=115, maxhr=175,
                    unk_chol=False, chol=180, fbs="No", ecg="Normal", oldpeak=0,
                    slope=list(SLOPE)[0])
    elif kind == "risk":
        vals = dict(age=62, sex="Male", cp=list(CP)[0], exang="Yes", bp=150, maxhr=105,
                    unk_chol=False, chol=270, fbs="Yes", ecg=list(ECG)[1], oldpeak=3,
                    slope=list(SLOPE)[1])
    else:
        vals = {k: v for k, v in DEFAULTS.items()}
    vals["checked"] = False
    for k, v in vals.items():
        st.session_state[k] = v


def explain(text):
    st.markdown(f'<div class="explain"><b>What this means:</b> {text}</div>', unsafe_allow_html=True)


def badge(text, kind):
    st.markdown(f'<span class="badge b-{kind}">{text}</span>', unsafe_allow_html=True)


# =====================================================================
#  SIDEBAR
# =====================================================================
with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown("""
1. Fill the **4 tabs** (Personal, Symptoms, Measurements, ECG)
2. Press **Check My Risk**
3. Read your **score and the reasons** behind it
""")
    st.subheader("⚡ Try an example")
    st.button("🟢 Healthy example", on_click=load_example, args=("healthy",), use_container_width=True)
    st.button("🔴 High-risk example", on_click=load_example, args=("risk",), use_container_width=True)
    st.button("↺ Reset", on_click=load_example, args=("reset",), use_container_width=True)
    st.subheader("📖 Quick glossary")
    st.markdown("""
- **Angina**: chest pain from low blood flow to the heart
- **ECG**: a test that records the heart's electrical signal
- **Max heart rate**: fastest heartbeat reached during exercise
- **Oldpeak**: how far the ECG line drops during exercise
""")
    st.caption("Model: Logistic Regression · Alert level: 30%")

# =====================================================================
#  HEADER
# =====================================================================
st.markdown("""
<div class="hero">
  <h1>❤️ Heart Health Checker</h1>
  <p>Answer a few simple questions and get an easy-to-understand heart risk estimate.</p>
</div>
<div class="steps">
  <div class="step"><b>1️⃣ Fill details</b>4 short tabs</div>
  <div class="step"><b>2️⃣ Press the button</b>Check My Risk</div>
  <div class="step"><b>3️⃣ Understand</b>Score + reasons</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["👤 1. Personal", "🩺 2. Symptoms", "🩸 3. Measurements", "📈 4. ECG"])

# ---------------- TAB 1: PERSONAL ----------------
with tab1:
    st.subheader("About you")
    c1, c2 = st.columns(2)
    c1.slider("Age (years)", 20, 90, key="age", help="Your age in years. Risk rises with age.")
    c2.radio("Sex", ["Male", "Female"], key="sex", horizontal=True,
             help="Biological sex. In this data, men showed a higher rate of heart disease.")
    age = st.session_state.age
    if age < 40:
        badge("Younger age group: usually lower baseline risk", "good")
    elif age < 55:
        badge("Middle age group: risk starts to rise", "warn")
    else:
        badge("Older age group: higher baseline risk", "bad")
    st.caption("➡️ Next: open the **Symptoms** tab.")

# ---------------- TAB 2: SYMPTOMS ----------------
with tab2:
    st.subheader("How do you feel?")
    st.selectbox("Chest pain type", list(CP), key="cp",
                 help="Choose the option that best matches your chest pain. If you have none, pick the first one.")
    explain(CP[st.session_state.cp][1])

    st.radio("Do you feel chest pain during exercise?", list(EXANG), key="exang", horizontal=True,
             help="For example while walking fast, running or climbing stairs.")
    explain(EXANG[st.session_state.exang])
    st.caption("➡️ Next: open the **Measurements** tab.")

# ---------------- TAB 3: MEASUREMENTS ----------------
with tab3:
    st.subheader("Your body numbers")
    c1, c2 = st.columns(2)
    c1.number_input("Resting blood pressure (mm Hg)", 80, 220, key="bp",
                    help="Top number of your BP reading, measured while resting.")
    c2.slider("Max heart rate (beats/min)", 60, 210, key="maxhr",
              help="Fastest heartbeat reached during exercise or a stress test.")

    bp = st.session_state.bp
    if bp < 120:
        badge("🟢 Blood pressure: Normal", "good")
    elif bp < 140:
        badge("🟡 Blood pressure: Slightly high", "warn")
    else:
        badge("🔴 Blood pressure: High", "bad")

    expected = 220 - age
    ratio = st.session_state.maxhr / expected
    if ratio >= 0.85:
        badge(f"🟢 Heart rate is strong (expected max for your age ≈ {expected})", "good")
    elif ratio >= 0.7:
        badge(f"🟡 Heart rate is a bit low (expected max for your age ≈ {expected})", "warn")
    else:
        badge(f"🔴 Heart rate is low for your age (expected max ≈ {expected})", "bad")

    st.checkbox("I don't know my cholesterol", key="unk_chol",
                help="We will use the dataset's typical value (237 mg/dl).")
    if st.session_state.unk_chol:
        st.info("Using the typical value **237 mg/dl**. Your result will be less personal.")
        chol = 237
    else:
        st.number_input("Cholesterol (mg/dl)", 80, 600, key="chol",
                        help="Total cholesterol from a blood test.")
        chol = st.session_state.chol
        if chol < 200:
            badge("🟢 Cholesterol: Desirable (below 200)", "good")
        elif chol < 240:
            badge("🟡 Cholesterol: Borderline (200 to 239)", "warn")
        else:
            badge("🔴 Cholesterol: High (240 or more)", "bad")

    st.radio("Fasting blood sugar above 120 mg/dl?", list(FBS), key="fbs", horizontal=True,
             help="Blood sugar measured after not eating for at least 8 hours.")
    explain(FBS[st.session_state.fbs])
    st.caption("➡️ Last step: open the **ECG** tab.")

# ---------------- TAB 4: ECG ----------------
with tab4:
    st.subheader("Heart test results")
    st.selectbox("Resting ECG result", list(ECG), key="ecg",
                 help="Result of the heart's electrical test taken while resting.")
    explain(ECG[st.session_state.ecg][1])

    st.slider("Oldpeak (ST depression)", 0, 6, key="oldpeak",
              help="How much the ECG line drops during exercise compared to rest. Whole numbers only.")
    op = st.session_state.oldpeak
    if op == 0:
        badge("🟢 No drop: normal", "good")
    elif op <= 2:
        badge("🟡 Mild drop", "warn")
    else:
        badge("🔴 Large drop: a strong warning sign", "bad")

    st.selectbox("ST slope during exercise", list(SLOPE), key="slope",
                 help="Direction of the ECG line at peak exercise.")
    explain(SLOPE[st.session_state.slope][1])
    st.caption("✅ All done! Press the button below.")

# =====================================================================
#  PREDICT
# =====================================================================
st.write("")
if st.button("🔍 Check My Risk", type="primary", use_container_width=True):
    st.session_state.checked = True

if st.session_state.checked:
    s = st.session_state
    cp_code, ecg_code, slope_code = CP[s.cp][0], ECG[s.ecg][0], SLOPE[s.slope][0]

    row = {
        "Age": s.age, "RestingBP": s.bp, "Cholesterol": chol,
        "FastingBS": 1 if s.fbs == "Yes" else 0, "MaxHR": s.maxhr, "Oldpeak": s.oldpeak,
        "Sex_M": 1 if s.sex == "Male" else 0,
        "ChestPainType_ATA": int(cp_code == "ATA"), "ChestPainType_NAP": int(cp_code == "NAP"),
        "ChestPainType_TA": int(cp_code == "TA"),
        "RestingECG_Normal": int(ecg_code == "Normal"), "RestingECG_ST": int(ecg_code == "ST"),
        "ExerciseAngina_Y": 1 if s.exang == "Yes" else 0,
        "ST_Slope_Flat": int(slope_code == "Flat"), "ST_Slope_Up": int(slope_code == "Up"),
    }
    X_new = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)
    scaled = scaler.transform(X_new)
    prob = float(model.predict_proba(scaled)[0][1])
    pct = round(prob * 100, 1)

    if prob < THRESHOLD:
        cls, color, icon = "r-low", "#2a9d5c", "✅"
        title = "Lower risk of heart disease"
        msg = "Your answers look reassuring. Keep up your healthy habits."
    elif prob < 0.6:
        cls, color, icon = "r-mid", "#d98200", "⚠️"
        title = "Moderate risk: worth a check-up"
        msg = "Some answers raised the score. A doctor visit is a good idea."
    else:
        cls, color, icon = "r-high", "#e63946", "🚨"
        title = "Higher risk of heart disease"
        msg = "Several answers raised the score a lot. Please see a doctor soon."

    st.markdown("## 📊 Your Result")
    st.markdown(f"""
    <div class="result {cls}">
      <p class="pct" style="color:{color};">{icon} {pct}%</p>
      <h3>{title}</h3>
      <p>{msg}</p>
    </div>
    <div class="meter">
      <div class="tick" style="left:{THRESHOLD*100}%;"></div>
      <div class="pin" style="left:{min(max(pct, 1), 99)}%;"></div>
    </div>
    <div class="meter-labels"><span>0% Low</span><span>Alert line: 30%</span><span>100% High</span></div>
    """, unsafe_allow_html=True)

    with st.expander("❓ What does this percentage mean?"):
        st.markdown(f"""
- The number is the model's **estimated chance** of heart disease for someone with your answers.
- **Below 30%**: shown as lower risk.
- **30% to 60%**: moderate. The alert line is set at 30% on purpose, so the model **catches more
  real patients** (a missed patient is worse than an extra check-up).
- **Above 60%**: higher risk.
- It is **not a diagnosis**. Only a doctor can confirm.
""")

    # ----- Why this result -----
    st.markdown("### 🔎 Why did you get this result?")
    st.caption("Compared with an average patient in the data, these answers pushed your score up or down.")

    coef = model.coef_[0]
    contrib = dict(zip(columns, coef * scaled[0]))
    groups = {
        "Age": ["Age"], "Blood pressure": ["RestingBP"], "Cholesterol": ["Cholesterol"],
        "Fasting sugar": ["FastingBS"], "Max heart rate": ["MaxHR"], "Oldpeak (ECG drop)": ["Oldpeak"],
        "Sex": ["Sex_M"],
        "Chest pain type": ["ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA"],
        "Resting ECG": ["RestingECG_Normal", "RestingECG_ST"],
        "Pain during exercise": ["ExerciseAngina_Y"],
        "ST slope": ["ST_Slope_Flat", "ST_Slope_Up"],
    }
    effects = {g: sum(contrib[c] for c in cols) for g, cols in groups.items()}
    top = sorted(effects.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
    biggest = max(abs(v) for _, v in top) or 1
    html = ""
    for name, val in top:
        w = max(4, int(abs(val) / biggest * 100))
        arrow, cl = ("⬆️ raises risk", "up") if val > 0 else ("⬇️ lowers risk", "down")
        html += (f'<div class="factor"><div class="name"><b>{name}</b><br><small>{arrow}</small></div>'
                 f'<div class="bar-wrap"><div class="bar {cl}" style="width:{w}%"></div></div></div>')
    st.markdown(html, unsafe_allow_html=True)
    st.caption("🔴 red bar = increases risk · 🟢 green bar = decreases risk")

    # ----- What to do next -----
    st.markdown("### ✅ What should I do next?")
    if prob < THRESHOLD:
        tips = ["Keep a regular routine: walk 30 minutes most days.",
                "Eat less oily and salty food.",
                "Do a routine health check once a year."]
    elif prob < 0.6:
        tips = ["Book an appointment with a doctor and show them these numbers.",
                "Ask about an ECG or a stress test.",
                "Track your blood pressure and cholesterol every few weeks."]
    else:
        tips = ["Please consult a doctor (preferably a cardiologist) soon.",
                "If you have chest pain, breathlessness or dizziness, go to a hospital immediately.",
                "Do not wait for symptoms to get worse."]
    for t in tips:
        st.markdown(f'<div class="tip">👉 {t}</div>', unsafe_allow_html=True)

    with st.expander("📋 See everything you entered"):
        st.table(pd.DataFrame({
            "Question": ["Age", "Sex", "Chest pain", "Pain during exercise", "Blood pressure",
                         "Max heart rate", "Cholesterol", "Sugar > 120", "ECG", "Oldpeak", "ST slope"],
            "Your answer": [s.age, s.sex, s.cp, s.exang, s.bp, s.maxhr,
                            f"{chol} (typical value)" if s.unk_chol else chol,
                            s.fbs, s.ecg, s.oldpeak, s.slope],
        }).astype(str))

st.markdown('<div class="footer">⚠️ Student learning project. This is <b>NOT</b> a medical diagnosis. '
            'Always consult a qualified doctor.</div>', unsafe_allow_html=True)

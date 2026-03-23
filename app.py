import streamlit as st
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Inter:wght@400;600;700;800&display=swap');

    .stApp {
        background-color: #fdf6f0;
        color: #3d2b2b;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 0px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Navbar */
    .navbar {
        background: linear-gradient(135deg, #d4829a, #e8a0b0, #a8c5a0);
        border-radius: 0 0 24px 24px;
        padding: 20px 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 32px;
        box-shadow: 0 4px 20px rgba(212,130,154,0.3);
    }
    .navbar-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .navbar-logo { font-size: 36px; }
    .navbar-title {
        font-family: 'Playfair Display', serif;
        font-size: 30px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .navbar-title span {
        font-style: italic;
        font-weight: 400;
        color: #fff0f5;
    }
    .navbar-subtitle {
        font-size: 13px;
        color: #fff0f5;
        margin-top: 3px;
        font-weight: 500;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255,255,255,0.25) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        gap: 4px !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px 28px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.4) !important;
        color: #7a3050 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        margin-top: -72px !important;
        margin-bottom: 32px !important;
        float: right !important;
        width: fit-content !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 8px !important;
    }

    /* Section headers */
    .section-header {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        color: #c0607a;
        font-size: 18px;
        font-weight: 400;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #fde8ef;
    }

    /* Verdict boxes */
    .verdict-setosa {
        background: linear-gradient(135deg, #f8d7e3, #fce4ec);
        border: 2px solid #e91e8c;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(233,30,140,0.15);
    }
    .verdict-versicolor {
        background: linear-gradient(135deg, #d4edda, #e8f5e9);
        border: 2px solid #4caf50;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(76,175,80,0.15);
    }
    .verdict-virginica {
        background: linear-gradient(135deg, #dde8f8, #e8eef8);
        border: 2px solid #5c7cfa;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(92,124,250,0.15);
    }

    /* Polaroid image card */
    .polaroid {
        background: #ffffff;
        border-radius: 16px;
        padding: 12px 12px 20px 12px;
        box-shadow: 0 8px 24px rgba(212,130,154,0.2);
        margin: 16px auto;
        max-width: 260px;
        text-align: center;
        border: 1px solid #f5d5e0;
    }
    .polaroid img {
        border-radius: 10px;
        width: 100%;
        height: 180px;
        object-fit: cover;
    }
    .polaroid-caption {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        color: #c0607a;
        font-size: 15px;
        margin-top: 10px;
        font-weight: 400;
    }

    /* Confidence badge */
    .confidence-badge {
        background-color: #fff0f5;
        color: #c0607a;
        padding: 6px 20px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 15px;
        border: 2px solid #e8a0b0;
        display: inline-block;
        margin-top: 8px;
    }

    /* Sliders */
    .stSlider label {
        color: #7a3050 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #d4829a, #c0607a);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-size: 17px;
        font-weight: 800;
        width: 100%;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(192,96,122,0.4);
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #c0607a, #a84060);
        transform: translateY(-2px);
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #fff0f5;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #f5d5e0;
    }
    [data-testid="stMetricLabel"] {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #c0607a !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 22px !important;
        font-weight: 800 !important;
        color: #7a3050 !important;
    }

    /* Petal rain */
    @keyframes petalFall {
        0%   { transform: translateY(-100px) rotate(0deg) translateX(0px); opacity:1; }
        50%  { transform: translateY(50vh) rotate(180deg) translateX(25px); opacity:0.8; }
        100% { transform: translateY(110vh) rotate(360deg) translateX(-15px); opacity:0; }
    }
    .petal-1  { position:fixed; left:3%;  top:-60px; font-size:20px; animation:petalFall 3.0s 0.0s linear forwards; z-index:9999; pointer-events:none; }
    .petal-2  { position:fixed; left:9%;  top:-60px; font-size:24px; animation:petalFall 3.5s 0.2s linear forwards; z-index:9999; pointer-events:none; }
    .petal-3  { position:fixed; left:16%; top:-60px; font-size:18px; animation:petalFall 2.8s 0.4s linear forwards; z-index:9999; pointer-events:none; }
    .petal-4  { position:fixed; left:23%; top:-60px; font-size:22px; animation:petalFall 3.2s 0.1s linear forwards; z-index:9999; pointer-events:none; }
    .petal-5  { position:fixed; left:30%; top:-60px; font-size:26px; animation:petalFall 3.8s 0.3s linear forwards; z-index:9999; pointer-events:none; }
    .petal-6  { position:fixed; left:37%; top:-60px; font-size:20px; animation:petalFall 2.9s 0.5s linear forwards; z-index:9999; pointer-events:none; }
    .petal-7  { position:fixed; left:44%; top:-60px; font-size:24px; animation:petalFall 3.4s 0.2s linear forwards; z-index:9999; pointer-events:none; }
    .petal-8  { position:fixed; left:51%; top:-60px; font-size:18px; animation:petalFall 3.1s 0.4s linear forwards; z-index:9999; pointer-events:none; }
    .petal-9  { position:fixed; left:58%; top:-60px; font-size:22px; animation:petalFall 2.7s 0.1s linear forwards; z-index:9999; pointer-events:none; }
    .petal-10 { position:fixed; left:65%; top:-60px; font-size:26px; animation:petalFall 3.6s 0.3s linear forwards; z-index:9999; pointer-events:none; }
    .petal-11 { position:fixed; left:72%; top:-60px; font-size:20px; animation:petalFall 3.0s 0.5s linear forwards; z-index:9999; pointer-events:none; }
    .petal-12 { position:fixed; left:79%; top:-60px; font-size:24px; animation:petalFall 3.3s 0.2s linear forwards; z-index:9999; pointer-events:none; }
    .petal-13 { position:fixed; left:86%; top:-60px; font-size:18px; animation:petalFall 2.8s 0.4s linear forwards; z-index:9999; pointer-events:none; }
    .petal-14 { position:fixed; left:93%; top:-60px; font-size:22px; animation:petalFall 3.5s 0.6s linear forwards; z-index:9999; pointer-events:none; }
    .petal-15 { position:fixed; left:6%;  top:-60px; font-size:20px; animation:petalFall 3.2s 0.7s linear forwards; z-index:9999; pointer-events:none; }
    .petal-16 { position:fixed; left:13%; top:-60px; font-size:26px; animation:petalFall 2.9s 0.8s linear forwards; z-index:9999; pointer-events:none; }
    .petal-17 { position:fixed; left:48%; top:-60px; font-size:22px; animation:petalFall 3.7s 0.9s linear forwards; z-index:9999; pointer-events:none; }
    .petal-18 { position:fixed; left:75%; top:-60px; font-size:18px; animation:petalFall 3.1s 1.0s linear forwards; z-index:9999; pointer-events:none; }
    .petal-19 { position:fixed; left:90%; top:-60px; font-size:24px; animation:petalFall 2.6s 0.3s linear forwards; z-index:9999; pointer-events:none; }
    .petal-20 { position:fixed; left:33%; top:-60px; font-size:20px; animation:petalFall 3.4s 0.5s linear forwards; z-index:9999; pointer-events:none; }
</style>
""", unsafe_allow_html=True)

# ── Petal rain HTML ─────────────────────────────────────────
petal_rain = """
<div class="petal-1">🌸</div>
<div class="petal-2">🌺</div>
<div class="petal-3">🌼</div>
<div class="petal-4">🌸</div>
<div class="petal-5">🌷</div>
<div class="petal-6">🌺</div>
<div class="petal-7">🌸</div>
<div class="petal-8">🌼</div>
<div class="petal-9">🌷</div>
<div class="petal-10">🌸</div>
<div class="petal-11">🌺</div>
<div class="petal-12">🌼</div>
<div class="petal-13">🌸</div>
<div class="petal-14">🌷</div>
<div class="petal-15">🌺</div>
<div class="petal-16">🌸</div>
<div class="petal-17">🌼</div>
<div class="petal-18">🌷</div>
<div class="petal-19">🌸</div>
<div class="petal-20">🌺</div>
"""

# ── Load model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load("iris_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# ── Flower data ─────────────────────────────────────────────
flower_info = {
    "Iris-setosa": {
        "image":       "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
        "emoji":       "🌸",
        "color":       "verdict-setosa",
        "desc":        "Small and delicate — found in arctic and subarctic regions.",
        "text_color":  "#880e4f",
        "fun_fact":    "Setosa petals are so small they are almost invisible!"
    },
    "Iris-versicolor": {
        "image":       "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "emoji":       "🪻",
        "color":       "verdict-versicolor",
        "desc":        "The blue flag iris — common across North America.",
        "text_color":  "#1b5e20",
        "fun_fact":    "Versicolor means 'variously colored' in Latin."
    },
    "Iris-virginica": {
        "image":       "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "emoji":       "💐",
        "color":       "verdict-virginica",
        "desc":        "The Virginia iris — largest of the three species.",
        "text_color":  "#1a237e",
        "fun_fact":    "Virginica can grow up to 80cm tall in the wild."
    }
}

# ── Navbar ───────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-left">
        <div class="navbar-logo">🌿</div>
        <div>
            <div class="navbar-title">
                Iris <span>Classifier</span>
            </div>
            <div class="navbar-subtitle">
                Identify Iris flower species instantly
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌿  Identify",
    "📊  Model Insights",
    "ℹ️   About"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — IDENTIFY
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-family:'Playfair Display',serif; font-style:italic;
                   color:#c0607a; font-size:32px;
                   font-weight:400; margin:0;">
            What flower is this?
        </h2>
        <p style="color:#a07080; font-size:16px; margin-top:6px;">
            Adjust the measurements below to identify the Iris species.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown("""
        <div style="background:#ffffff; border-radius:16px;
                    padding:12px 20px 2px 20px; border:1px solid #f5d5e0;
                    box-shadow:0 2px 8px rgba(212,130,154,0.1);
                    margin-bottom:0px;">
            <div class="section-header" style="margin-bottom:4px;">🌿 Sepal Measurements</div>
        </div>
        """, unsafe_allow_html=True)

        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0, max_value=8.0,
            value=5.1, step=0.1
        )
        st.caption(f"Selected: **{sepal_length} cm**")

        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0, max_value=5.0,
            value=3.5, step=0.1
        )
        st.caption(f"Selected: **{sepal_width} cm**")

        # Petal measurements — FIX: removed the unclosed opening <div> wrapper
        st.markdown("""
        <div style="background:#ffffff; border-radius:16px;
                    padding:12px 20px 2px 20px; border:1px solid #f5d5e0;
                    box-shadow:0 2px 8px rgba(212,130,154,0.1);
                    margin-bottom:0px;">
            <div class="section-header" style="margin-bottom:4px;">🌸 Petal Measurements</div>
        </div>
        """, unsafe_allow_html=True)

        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0, max_value=7.0,
            value=1.4, step=0.1
        )
        st.caption(f"Selected: **{petal_length} cm**")

        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1, max_value=2.5,
            value=0.2, step=0.1
        )
        st.caption(f"Selected: **{petal_width} cm**")

        predict_btn = st.button("🌸  Identify Species")

    # ── Results ───────────────────────────────────────────────
    with col_right:
        if predict_btn:
            # Petal rain
            st.markdown(petal_rain, unsafe_allow_html=True)

            # Predict
            input_data    = np.array([[sepal_length, sepal_width,
                                        petal_length, petal_width]])
            prediction    = model.predict(input_data)
            probabilities = model.predict_proba(input_data)
            species       = encoder.inverse_transform(prediction)[0]
            confidence    = round(np.max(probabilities) * 100, 2)

            info = flower_info.get(species, {
                "image": "", "emoji": "🌸",
                "color": "verdict-setosa",
                "desc": "", "text_color": "#880e4f",
                "fun_fact": ""
            })

            # Verdict
            st.markdown(f"""
            <div class="{info['color']}">
                <div style="font-size:48px; margin-bottom:8px;">
                    {info['emoji']}
                </div>
                <div style="font-family:'Playfair Display',serif;
                            font-style:italic;
                            color:{info['text_color']}; font-size:13px;
                            font-weight:400; letter-spacing:2px;">
                    identified species
                </div>
                <div style="font-family:'Playfair Display',serif;
                            color:{info['text_color']}; font-size:26px;
                            font-weight:700; margin:8px 0;">
                    {species}
                </div>
                <div style="color:{info['text_color']}; font-size:14px;
                            opacity:0.8; margin-bottom:8px;">
                    {info['desc']}
                </div>
                <span class="confidence-badge">
                    {confidence:.2f}% confidence
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Polaroid image
            try:
                st.markdown(f"""
                <div style="background:#ffffff; border-radius:16px;
                            padding:12px 12px 4px 12px;
                            box-shadow:0 8px 24px rgba(212,130,154,0.2);
                            margin:16px auto; max-width:260px;
                            text-align:center; border:1px solid #f5d5e0;">
                    <img src="{info['image']}" style="border-radius:10px; width:100%; height:180px; object-fit:cover;" />
                    <div class="polaroid-caption">{species}</div>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.markdown(f"""
                <div style="text-align:center; padding:20px; font-size:60px;">
                    {info['emoji']}
                </div>
                """, unsafe_allow_html=True)

            # Fun fact
            st.markdown(f"""
            <div style="background:#fff8f0; border-radius:12px;
                        padding:16px 20px; border-left:4px solid #e8a0b0;
                        margin:16px 0;">
                <div style="font-family:'Playfair Display',serif;
                            font-style:italic; color:#c0607a;
                            font-size:13px; margin-bottom:4px;">
                    did you know?
                </div>
                <div style="color:#7a3050; font-size:14px;
                            font-weight:600;">
                    {info['fun_fact']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Measurements summary
            st.markdown('<div class="section-header">📋 Your Measurements</div>',
                        unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Sepal Length", f"{sepal_length} cm")
                st.metric("Petal Length", f"{petal_length} cm")
            with m2:
                st.metric("Sepal Width",  f"{sepal_width} cm")
                st.metric("Petal Width",  f"{petal_width} cm")

        else:
            st.markdown("""
            <div style="display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        height:420px; text-align:center;">
                <div style="font-size:72px; margin-bottom:16px;">🌿</div>
                <div style="font-family:'Playfair Display',serif;
                            font-style:italic;
                            color:#c0607a; font-size:26px;
                            font-weight:400; margin-bottom:8px;">
                    What flower is this?
                </div>
                <div style="color:#c4a0b0; font-size:15px;
                            max-width:260px; line-height:1.7;">
                    Adjust the measurements and click
                    <strong>Identify Species</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-family:'Playfair Display',serif; font-style:italic;
                   color:#c0607a; font-size:32px;
                   font-weight:400; margin:0;">
            How does it decide?
        </h2>
        <p style="color:#a07080; font-size:16px; margin-top:6px;">
            Understanding what the model learned from the data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    import plotly.graph_objects as go

    # Feature importance
    st.markdown('<div class="section-header">🌱 Feature Importance</div>',
                unsafe_allow_html=True)

    feature_names = ["Sepal Length", "Sepal Width",
                     "Petal Length", "Petal Width"]
    importances   = model.feature_importances_
    colors        = ['#e8a0b0', '#a8c5a0', '#d4829a', '#7db87d']

    fig = go.Figure(go.Bar(
        x=feature_names,
        y=importances,
        marker_color=colors,
        text=[f'{v:.3f}' for v in importances],
        textposition='outside',
        textfont=dict(size=13, color='#7a3050')
    ))
    fig.update_layout(
        height=360,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#7a3050', size=13),
        yaxis=dict(gridcolor='#fde8ef', title='Importance Score'),
        xaxis=dict(gridcolor='#fde8ef'),
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # Species cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🌺 Species Reference</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (species, info) in zip(
        [c1, c2, c3], flower_info.items()
    ):
        with col:
            st.markdown(f"""
            <div style="background:#ffffff; border-radius:16px;
                        padding:20px; border:1px solid #f5d5e0;
                        text-align:center;
                        box-shadow:0 2px 8px rgba(212,130,154,0.12);">
                <div style="font-size:40px; margin-bottom:8px;">
                    {info['emoji']}
                </div>
                <div style="font-family:'Playfair Display',serif;
                            font-style:italic;
                            font-weight:700;
                            color:{info['text_color']};
                            font-size:16px; margin:8px 0;">
                    {species}
                </div>
                <div style="color:#a07080; font-size:13px;
                            line-height:1.5; margin-bottom:8px;">
                    {info['desc']}
                </div>
                <div style="color:#c0607a; font-size:12px;
                            font-style:italic; background:#fff0f5;
                            border-radius:8px; padding:8px;">
                    {info['fun_fact']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="margin-bottom: 32px;">
        <h2 style="font-family:'Playfair Display',serif; font-style:italic;
                   color:#c0607a; font-size:32px;
                   font-weight:400; margin:0;">
            About Iris Classifier
        </h2>
        <p style="color:#a07080; font-size:16px; margin-top:6px;">
            A machine learning app for Iris flower species identification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div style="background:#ffffff; border-radius:16px;
                    padding:28px; border:1px solid #f5d5e0;
                    box-shadow:0 2px 8px rgba(212,130,154,0.12);
                    margin-bottom:20px;">
            <div class="section-header">🌿 The Dataset</div>
            <p style="color:#7a3050; font-size:15px; line-height:1.8;">
                The classic Iris dataset contains <strong>150 samples</strong>
                across 3 species. Each sample has 4 measurements —
                sepal length, sepal width, petal length and petal width.
                First introduced by Ronald Fisher in 1936.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#ffffff; border-radius:16px;
                    padding:28px; border:1px solid #f5d5e0;
                    box-shadow:0 2px 8px rgba(212,130,154,0.12);">
            <div class="section-header">🛠️ Tech Stack</div>
            <div style="display:flex; flex-wrap:wrap; gap:10px;">
                <span style="background:#fde8ef; color:#c0607a;
                             padding:6px 16px; border-radius:20px;
                             font-weight:700; font-size:14px;">Python</span>
                <span style="background:#fde8ef; color:#c0607a;
                             padding:6px 16px; border-radius:20px;
                             font-weight:700; font-size:14px;">Scikit-learn</span>
                <span style="background:#fde8ef; color:#c0607a;
                             padding:6px 16px; border-radius:20px;
                             font-weight:700; font-size:14px;">Streamlit</span>
                <span style="background:#fde8ef; color:#c0607a;
                             padding:6px 16px; border-radius:20px;
                             font-weight:700; font-size:14px;">Plotly</span>
                <span style="background:#fde8ef; color:#c0607a;
                             padding:6px 16px; border-radius:20px;
                             font-weight:700; font-size:14px;">NumPy</span>
                <span style="background:#fde8ef; color:#c0607a;
                             padding:6px 16px; border-radius:20px;
                             font-weight:700; font-size:14px;">Joblib</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#ffffff; border-radius:16px;
                    padding:28px; border:1px solid #f5d5e0;
                    box-shadow:0 2px 8px rgba(212,130,154,0.12);
                    margin-bottom:20px;">
            <div class="section-header">🌺 How it works</div>
            <p style="color:#7a3050; font-size:15px; line-height:1.8;">
                A <strong>Random Forest classifier</strong> was trained
                on the Iris dataset. Petal measurements are the strongest
                predictors — petal length alone can distinguish Setosa
                from the other two species with near perfect accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#ffffff; border-radius:16px;
                    padding:28px; border:1px solid #f5d5e0;
                    box-shadow:0 2px 8px rgba(212,130,154,0.12);">
            <div class="section-header">📊 Key Stats</div>
            <div style="display:flex; flex-direction:column; gap:0;">
                <div style="display:flex; justify-content:space-between;
                            padding:12px 0;
                            border-bottom:1px solid #fde8ef;">
                    <span style="color:#a07080; font-size:14px;">
                        Training samples
                    </span>
                    <span style="color:#c0607a; font-weight:800;">150</span>
                </div>
                <div style="display:flex; justify-content:space-between;
                            padding:12px 0;
                            border-bottom:1px solid #fde8ef;">
                    <span style="color:#a07080; font-size:14px;">
                        Species classified
                    </span>
                    <span style="color:#c0607a; font-weight:800;">3</span>
                </div>
                <div style="display:flex; justify-content:space-between;
                            padding:12px 0;
                            border-bottom:1px solid #fde8ef;">
                    <span style="color:#a07080; font-size:14px;">
                        Features used
                    </span>
                    <span style="color:#c0607a; font-weight:800;">4</span>
                </div>
                <div style="display:flex; justify-content:space-between;
                            padding:12px 0;">
                    <span style="color:#a07080; font-size:14px;">
                        Strongest predictor
                    </span>
                    <span style="color:#c0607a; font-weight:800;">
                        Petal length
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#c4a0b0; font-size:13px;
                font-family:'Playfair Display',serif; font-style:italic;">
        🌸 Built by Arjuman Sultana &nbsp;·&nbsp;
        Machine Learning · Streamlit · Python
    </div>
    """, unsafe_allow_html=True)
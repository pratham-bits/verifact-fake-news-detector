"""
VeriFact — AI-Powered Fake News Detector
Streamlit App | app.py

Local run:
    streamlit run app.py

Deployment:
    Add GEMINI_API_KEY to Streamlit Cloud secrets
"""

import os
import re
import json
import pickle
import warnings
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="VeriFact — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

# ── Resolve paths relative to this file ─────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# ── Load resources ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(os.path.join(MODELS_DIR, 'best_model.pkl'), 'rb') as f:
        pipeline = pickle.load(f)
    with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)
    return pipeline, metadata


@st.cache_resource
def load_nlp():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import spacy
    nlp   = spacy.load('en_core_web_sm', disable=['parser'])
    vader = SentimentIntensityAnalyzer()
    return nlp, vader


@st.cache_resource
def load_gemini():
    from google import genai
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    client = genai.Client(api_key=api_key)
    return client


# ── Helper functions ─────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def get_sentiment(text, vader):
    scores   = vader.polarity_scores(str(text))
    compound = scores['compound']
    label    = 'Positive' if compound >= 0.05 else ('Negative' if compound <= -0.05 else 'Neutral')
    return {
        'compound': round(compound, 4),
        'label':    label,
        'positive': round(scores['pos'], 4),
        'negative': round(scores['neg'], 4)
    }


def get_tone_score(text):
    text_str = str(text)
    words    = text_str.split()
    total    = max(len(words), 1)
    excl  = min(text_str.count('!') * 5, 30)
    ques  = min(text_str.count('?') * 3, 15)
    caps  = min(sum(1 for w in words if w.isupper() and len(w) > 2) / total * 100, 25)
    SENS  = ['breaking','shocking','unbelievable','exposed','secret','banned',
             'conspiracy','hoax','fraud','urgent','alert','share before',
             'before its deleted','wake up','they dont want','mainstream media']
    kw    = min(sum(1 for k in SENS if k in text_str.lower()) * 5, 30)
    return min(round(excl + ques + caps + kw, 1), 100)


def extract_entities(text, nlp_model, max_chars=2000):
    doc      = nlp_model(str(text)[:max_chars])
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    return entities


def detect_language(text):
    from langdetect import detect, LangDetectException
    try:
        return detect(str(text)[:500])
    except LangDetectException:
        return 'unknown'


def call_gemini_safe(client, prompt, expect_json=False):
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={'temperature': 0.2, 'max_output_tokens': 400}
        )
        text = response.text.strip()
        if expect_json:
            text = re.sub(r'```json|```', '', text).strip()
            return json.loads(text)
        return text

    except json.JSONDecodeError:
        return {
            'verdict': 'Uncertain', 'confidence': 50,
            'explanation': 'Could not parse Gemini response.',
            'signals': [], 'verify_tip': 'Check a trusted news source.'
        }
    except Exception as e:
        err = str(e)
        if '429' in err or 'quota' in err.lower():
            return {
                'verdict': 'Uncertain', 'confidence': 50,
                'explanation': 'Gemini API quota exceeded for today. Showing ML result only. Quota resets daily.',
                'signals': [], 'verify_tip': 'Cross-check with Reuters, AP, PTI, or The Hindu.'
            }
        return {
            'error': True, 'verdict': 'Uncertain', 'confidence': 50,
            'explanation': f'Gemini unavailable: {err[:100]}',
            'signals': [], 'verify_tip': 'Check a trusted news source.'
        }


def full_predict(article_text, ml_pipeline, ml_metadata, nlp_model, vader_model, gemini_client):
    threshold = ml_metadata.get('confidence_threshold', 0.75)

    lang      = detect_language(article_text)
    sentiment = get_sentiment(article_text, vader_model)
    tone      = get_tone_score(article_text)
    ents      = extract_entities(article_text, nlp_model)
    nlp_data  = {
        'language':   lang,
        'is_english': lang == 'en',
        'sentiment':  sentiment,
        'tone_score': tone,
        'entities': {
            'people': ents.get('PERSON', [])[:5],
            'orgs':   ents.get('ORG',    [])[:5],
            'places': ents.get('GPE',    [])[:5],
            'dates':  ents.get('DATE',   [])[:3]
        }
    }

    result = {
        'nlp': nlp_data, 'ml_used': False, 'gemini_used': False,
        'verdict': None, 'confidence': None,
        'explanation': None, 'signals': [], 'verify_tip': None
    }

    gemini_available = gemini_client is not None

    # ── Non-English → Gemini full analysis ──────────────────────────
    if not nlp_data['is_english']:
        if gemini_available:
            prompt = f"""You are VeriFact, a multilingual AI fact-checker for students.
Analyze this article (any language). Translate if needed, then classify.
Do NOT answer Uncertain. Always choose Fake or Real.
Professional journalism with named sources = Real.
Sensational unverified claims = Fake.
Respond ONLY as JSON (no markdown):
{{"verdict":"Fake" or "Real","confidence":0-100,
"explanation":"2-3 sentences in English","signals":["s1","s2"],"verify_tip":"one tip"}}
Article: \"\"\"{article_text[:1200]}\"\"\""""
            gr = call_gemini_safe(gemini_client, prompt, expect_json=True)
            result.update({
                'gemini_used': True,
                'verdict':     gr.get('verdict', 'Uncertain'),
                'confidence':  gr.get('confidence', 50),
                'explanation': gr.get('explanation', ''),
                'signals':     gr.get('signals', []),
                'verify_tip':  gr.get('verify_tip', '')
            })
        else:
            result.update({
                'verdict':     'Uncertain',
                'confidence':  0,
                'explanation': 'Non-English article detected. Gemini API unavailable for multilingual analysis.',
                'verify_tip':  'Please verify with a trusted local news source.'
            })
        return result

    # ── English → ML model ───────────────────────────────────────────
    cleaned         = clean_text(article_text)
    predicted_class = ml_pipeline.predict([cleaned])[0]

    # WELFake confirmed encoding: 1 = Fake, 0 = Real
    ml_verdict    = 'Fake' if predicted_class == 1 else 'Real'
    score         = ml_pipeline.decision_function([cleaned])[0]
    confidence    = min(float(abs(score) / (abs(score) + 1)), 0.95)
    ml_confidence = confidence * 100

    result.update({
        'ml_used':    True,
        'verdict':    ml_verdict,
        'confidence': round(ml_confidence, 1)
    })

    # ── Low confidence → Gemini second opinion ───────────────────────
    if confidence < threshold:
        if gemini_available:
            prompt = f"""You are VeriFact. ML model was uncertain ({ml_verdict}, {ml_confidence:.1f}%).
Analyze independently. Do NOT answer Uncertain. Always choose Fake or Real.
Professional journalism with named sources and neutral tone = Real.
Sensational language, unverified claims, emotional manipulation = Fake.
Respond ONLY as JSON (no markdown):
{{"verdict":"Fake" or "Real","confidence":0-100,
"explanation":"2-3 sentences","signals":["s1","s2"],"verify_tip":"one specific tip"}}
Article: \"\"\"{article_text[:1000]}\"\"\""""
            gr = call_gemini_safe(gemini_client, prompt, expect_json=True)
            result.update({
                'gemini_used': True,
                'verdict':     gr.get('verdict', ml_verdict),
                'confidence':  gr.get('confidence', ml_confidence),
                'explanation': gr.get('explanation', ''),
                'signals':     gr.get('signals', []),
                'verify_tip':  gr.get('verify_tip', '')
            })
        else:
            # No Gemini — don't trust low confidence ML verdict
            result.update({
                'verdict':     'Uncertain',
                'confidence':  round(ml_confidence, 1),
                'explanation': (
                    f'The ML model had low confidence ({ml_confidence:.1f}%) on this article. '
                    f'Neutral, professionally-written articles often have weak TF-IDF signal. '
                    f'Add your Gemini API key in .streamlit/secrets.toml for a full analysis.'
                ),
                'verify_tip': 'Cross-check with Reuters, AP, PTI, or The Hindu.'
            })

    # ── High confidence → Gemini explanation only ────────────────────
    else:
        if gemini_available:
            prompt = f"""You are VeriFact, helping students understand news credibility.
ML verdict: {ml_verdict} ({ml_confidence:.1f}% confidence)
Tone score: {tone}/100 | Sentiment: {sentiment['label']} ({sentiment['compound']})
Article: \"\"\"{article_text[:800]}\"\"\"
Write 3-4 sentences explaining why this article is likely {ml_verdict}.
Point out 2-3 specific signals. End with one verification tip. English only."""
            explanation = call_gemini_safe(gemini_client, prompt, expect_json=False)
            result.update({
                'gemini_used': True,
                'explanation': explanation,
                'verify_tip':  'Cross-check with Reuters, AP, PTI, or The Hindu.'
            })
        else:
            result.update({
                'explanation': f'This article shows strong patterns consistent with {ml_verdict} news based on vocabulary and writing style analysis.',
                'verify_tip':  'Cross-check with Reuters, AP, PTI, or The Hindu.'
            })

    return result


# ── Custom CSS (your existing dark mode CSS preserved fully) ─────────
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117 !important;
        color: #e0e0e0 !important;
    }
    .main .block-container {
        background-color: #0e1117 !important;
        color: #e0e0e0 !important;
        padding: 2rem 1rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }
    .centered-heading { text-align: center !important; padding: 1rem 0 0.5rem 0 !important; }
    .centered-heading h1 { font-size: 3rem !important; color: #ffd700 !important; font-weight: 700 !important; margin-bottom: 0.2rem !important; }
    .centered-heading h3 { font-size: 1.3rem !important; color: #b0b0b0 !important; font-weight: 400 !important; margin-top: 0 !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffd700 !important; font-weight: 600 !important; }
    p, li, span, div, label, .stMarkdown, .stText, .stCaption { color: #e0e0e0 !important; }
    .stTextArea textarea, .stTextInput input {
        color: #ffffff !important; background-color: #1e1e1e !important;
        border: 2px solid #444444 !important; border-radius: 8px !important;
        font-size: 16px !important; padding: 12px !important; caret-color: #ffd700 !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder { color: #888888 !important; }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #ffd700 !important;
        box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.2) !important;
    }
    .verdict-fake {
        background-color: #2a1a1a !important; color: #ff8a80 !important;
        padding: 20px !important; border-radius: 12px !important;
        border-left: 6px solid #ff1744 !important; margin: 10px 0 !important;
    }
    .verdict-real {
        background-color: #1a2a1a !important; color: #b9f6ca !important;
        padding: 20px !important; border-radius: 12px !important;
        border-left: 6px solid #00e676 !important; margin: 10px 0 !important;
    }
    .verdict-uncertain {
        background-color: #2a2a1a !important; color: #ffab40 !important;
        padding: 20px !important; border-radius: 12px !important;
        border-left: 6px solid #ff9100 !important; margin: 10px 0 !important;
    }
    .stButton > button {
        background-color: #2d2d2d !important; color: #e0e0e0 !important;
        border-radius: 8px !important; border: 2px solid #555555 !important;
        font-weight: 600 !important; transition: all 0.3s ease !important;
        padding: 10px 24px !important; width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #3d3d3d !important; color: #ffd700 !important;
        border-color: #ffd700 !important; transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 215, 0, 0.2) !important;
    }
    .entity-tag {
        display: inline-block !important; background-color: #1a237e !important;
        color: #90caf9 !important; padding: 4px 14px !important;
        border-radius: 20px !important; margin: 4px !important; font-size: 13px !important;
        border: 1px solid #3f51b5 !important;
    }
    .signal-item {
        padding: 10px 16px !important; margin: 8px 0 !important;
        background-color: #1e1e1e !important; border-radius: 8px !important;
        border-left: 4px solid #ffd700 !important; color: #e0e0e0 !important;
    }
    .stMetric { background-color: #1a1a1a !important; padding: 15px !important; border-radius: 10px !important; border: 1px solid #333333 !important; }
    .stMetric .stMetricLabel { color: #b0b0b0 !important; }
    .stMetric .stMetricValue { color: #ffd700 !important; font-size: 1.8rem !important; }
    .css-1d391kg, .css-1lcbmhc, .stSidebar { background-color: #0a0a0f !important; border-right: 1px solid #2a2a2a !important; }
    hr { border-color: #333333 !important; margin: 1.5rem 0 !important; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #1a1a1a; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: #444444; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #ffd700; }
    @media screen and (max-width: 768px) {
        .main .block-container { padding: 1rem !important; }
        .centered-heading h1 { font-size: 2rem !important; }
        .stMetric .stMetricValue { font-size: 1.4rem !important; }
    }
    @media screen and (max-width: 480px) {
        .centered-heading h1 { font-size: 1.6rem !important; }
        .stButton > button { padding: 6px 12px !important; font-size: 12px !important; }
    }
    </style>
""", unsafe_allow_html=True)


# ── Session state init ───────────────────────────────────────────────
if 'article_text' not in st.session_state:
    st.session_state['article_text'] = ''


# ── Example button callbacks ─────────────────────────────────────────
def set_fake_example():
    st.session_state['article_text'] = (
        "BREAKING!! Scientists CONFIRM vaccines cause autism!! "
        "The government doesn't want you to know this secret! "
        "Obama and Hillary Clinton are involved in a massive CONSPIRACY. "
        "Share before they DELETE THIS!! WAKE UP AMERICA!!"
    )

def set_real_example():
    st.session_state['article_text'] = (
        "The Reserve Bank of India on Friday kept its benchmark repo rate unchanged "
        "at 6.5 per cent for the seventh consecutive time, as the six-member Monetary "
        "Policy Committee voted 5-1 to hold rates. Governor Shaktikanta Das said the "
        "committee remains focused on withdrawal of accommodation to ensure that "
        "inflation progressively aligns with the 4 per cent target while supporting growth."
    )


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/news.png", width=60)
    st.title("VeriFact")
    st.caption("AI-Powered Fake News Detector for Students")
    st.divider()

    st.subheader("📊 Model Info")
    try:
        _, metadata = load_model()
        st.success(f"**{metadata.get('model_name', 'Unknown')}**")
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", f"{metadata.get('accuracy', 0)*100:.1f}%")
        col_b.metric("ROC-AUC",  f"{metadata.get('roc_auc',  0)*100:.1f}%")
        st.caption("Trained on WELFake — 71,979 articles")
    except Exception:
        st.warning("Model files not found.")

    st.divider()
    st.subheader("ℹ️ How it works")
    st.markdown("""
1. 🌍 **Language detection**
2. 🤖 **ML classifier** (English)
3. 📊 **NLP analysis** (tone, sentiment, entities)
4. 💬 **Gemini AI** (explanation + multilingual)
    """)
    st.divider()
    st.caption("⚠️ Trained on US English news. Always verify with trusted sources.")


# ── Main content ─────────────────────────────────────────────────────
st.markdown("""
    <div class="centered-heading">
        <h1>🔍 VeriFact</h1>
        <h2>AI-Powered Fake News Detector for Students</h2>
    </div>
""", unsafe_allow_html=True)

st.divider()

article_input = st.text_area(
    "📰 Paste your news article here",
    height=200,
    placeholder="Paste the full article text or headline + first few paragraphs...",
    key='article_text'
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    analyze_btn = st.button(
        "🔍 Analyze Article", type="primary", use_container_width=True)
with col2:
    st.button("📌 Fake Example", on_click=set_fake_example, use_container_width=True)
with col3:
    st.button("📌 Real Example", on_click=set_real_example, use_container_width=True)

# ── Analysis ─────────────────────────────────────────────────────────
if analyze_btn and article_input.strip():
    if len(article_input.strip()) < 30:
        st.warning("Please enter at least 30 characters for a meaningful analysis.")
        st.stop()

    with st.spinner("Analyzing article... (this may take a few seconds)"):
        try:
            ml_pipeline, ml_metadata = load_model()
            nlp_model, vader_model   = load_nlp()
            gemini_client            = load_gemini()

            # Show progress steps
            status = st.empty()
            status.caption("🤖 Running ML classifier...")
            
            result = full_predict(
                article_input, ml_pipeline, ml_metadata,
                nlp_model, vader_model, gemini_client
            )
            status.empty()  # clear status message when done
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

    verdict    = result.get('verdict', 'Uncertain')
    confidence = result.get('confidence', 0)
    nlp_data   = result.get('nlp', {})
    sentiment  = nlp_data.get('sentiment', {})
    tone       = nlp_data.get('tone_score', 0)

    verdict_class = {
        'Fake': 'verdict-fake',
        'Real': 'verdict-real'
    }.get(verdict, 'verdict-uncertain')

    verdict_emoji = {
        'Fake': '🚨', 'Real': '✅', 'Uncertain': '⚠️'
    }.get(verdict, '❓')

    via_label = (
        'ML + Gemini'   if result.get('ml_used') and result.get('gemini_used') else
        'Gemini Only'   if result.get('gemini_used') else
        'ML Model Only'
    )

    st.markdown(f"""
    <div class="{verdict_class}">
        <h2 style="margin:0">{verdict_emoji} {verdict} News</h2>
        <p style="margin:0.3rem 0 0 0; font-size:1.1rem">
            Confidence: <strong>{confidence}%</strong> &nbsp;|&nbsp;
            Language: <strong>{nlp_data.get('language','en').upper()}</strong> &nbsp;|&nbsp;
            Analysis: <strong>{via_label}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence",      f"{confidence}%")
    c2.metric("Tone Score",      f"{tone}/100",
              help="0 = calm, 100 = very sensational")
    c3.metric("Sentiment",       sentiment.get('label', 'N/A'))
    c4.metric("Sentiment Score", f"{sentiment.get('compound', 0):.2f}")

    if result.get('explanation'):
        st.divider()
        st.subheader("📝 AI Explanation")
        st.info(result['explanation'])

    if result.get('signals'):
        st.subheader("🔎 Detection Signals")
        for signal in result['signals']:
            st.markdown(
                f'<div class="signal-item">• {signal}</div>',
                unsafe_allow_html=True)

    if result.get('verify_tip'):
        st.subheader("💡 Verification Tip")
        st.success(result['verify_tip'])

    entities = nlp_data.get('entities', {})
    if any(entities.values()):
        st.divider()
        st.subheader("🧩 Named Entities Found")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.write("**👤 People**")
            for p in entities.get('people', []):
                st.markdown(f'<span class="entity-tag">{p}</span>', unsafe_allow_html=True)
        with e2:
            st.write("**🏢 Organizations**")
            for o in entities.get('orgs', []):
                st.markdown(f'<span class="entity-tag">{o}</span>', unsafe_allow_html=True)
        with e3:
            st.write("**📍 Places**")
            for pl in entities.get('places', []):
                st.markdown(f'<span class="entity-tag">{pl}</span>', unsafe_allow_html=True)
        with e4:
            st.write("**📅 Dates**")
            for d in entities.get('dates', []):
                st.markdown(f'<span class="entity-tag">{d}</span>', unsafe_allow_html=True)

    with st.expander("🔧 Technical Details"):
        st.json({
            'model':       ml_metadata.get('model_name', 'N/A') if result.get('ml_used') else 'N/A',
            'confidence':  f"{confidence}%",
            'gemini_used': result.get('gemini_used'),
            'language':    nlp_data.get('language'),
            'tone_score':  tone,
            'sentiment':   sentiment
        })

elif analyze_btn:
    st.warning("Please paste an article before analyzing.")

st.divider()
st.caption(
    "⚠️ Always verify with trusted sources: Reuters, AP News, PTI, The Hindu, NDTV."
)
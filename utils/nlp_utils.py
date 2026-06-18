import re
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect, LangDetectException

nlp   = spacy.load("en_core_web_sm", disable=["parser"])
vader = SentimentIntensityAnalyzer()

def detect_language(text):
    try:
        return detect(str(text)[:500])
    except LangDetectException:
        return "unknown"

def get_sentiment(text):
    scores   = vader.polarity_scores(str(text))
    compound = scores["compound"]
    label    = "Positive" if compound >= 0.05 else ("Negative" if compound <= -0.05 else "Neutral")
    return {"compound": round(compound, 4), "label": label,
            "positive": round(scores["pos"], 4),
            "negative": round(scores["neg"], 4),
            "neutral":  round(scores["neu"], 4)}

def get_tone_score(text):
    text_str = str(text)
    words    = text_str.split()
    total    = max(len(words), 1)
    excl_score    = min(text_str.count("!") * 5, 30)
    ques_score    = min(text_str.count("?") * 3, 15)
    caps_score    = min(sum(1 for w in words if w.isupper() and len(w) > 2) / total * 100, 25)
    SENSATIONAL_WORDS = ["breaking","shocking","unbelievable","exposed","secret",
        "banned","conspiracy","hoax","fraud","urgent","alert","confirmed",
        "truth","mainstream media","share before","before its deleted","wake up"]
    keyword_score = min(sum(1 for kw in SENSATIONAL_WORDS if kw in text_str.lower()) * 5, 30)
    return min(round(excl_score + ques_score + caps_score + keyword_score, 1), 100)

def extract_entities(text, max_chars=2000):
    doc = nlp(str(text)[:max_chars])
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    return entities

def analyze_article(text):
    lang      = detect_language(text)
    sentiment = get_sentiment(text)
    tone      = get_tone_score(text)
    entities  = extract_entities(text)
    return {
        "language":   lang,
        "is_english": lang == "en",
        "sentiment":  sentiment,
        "tone_score": tone,
        "entities":   {
            "people": entities.get("PERSON", [])[:5],
            "orgs":   entities.get("ORG",    [])[:5],
            "places": entities.get("GPE",    [])[:5],
            "dates":  entities.get("DATE",   [])[:3]
        }
    }
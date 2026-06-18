import re
import json
import time
import google.generativeai as genai

GEMINI_MODEL = "gemini-2.0-flash"

def init_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)

def call_gemini(model, prompt, expect_json=False, retries=1):
    for attempt in range(retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2, max_output_tokens=400)
            )
            text = response.text.strip()
            if expect_json:
                text = re.sub(r"```json|```", "", text).strip()
                return json.loads(text)
            return text

        except json.JSONDecodeError:
            return {
                "verdict": "Uncertain", "confidence": 50,
                "explanation": "Could not parse Gemini response.",
                "signals": [], "verify_tip": "Check a trusted news source."
            }

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                if attempt < retries:
                    print(f"[RATE LIMIT] Waiting 60s before retry...")
                    time.sleep(60)
                    continue
                return {
                    "verdict": "Uncertain", "confidence": 50,
                    "explanation": "Gemini API quota exceeded. Try again later.",
                    "signals": [], "verify_tip": "Check a trusted news source."
                }
            if attempt < retries:
                time.sleep(3)
                continue
            return {
                "error": True, "verdict": "Uncertain", "confidence": 50,
                "explanation": "Gemini temporarily unavailable.",
                "signals": [], "verify_tip": "Check a trusted news source."
            }

    return {
        "verdict": "Uncertain", "confidence": 50,
        "explanation": "Max retries reached.",
        "signals": [], "verify_tip": "Check a trusted news source."
    }


def get_explanation(model, article_text, ml_verdict, ml_confidence, tone_score, sentiment):
    prompt = f"""
You are VeriFact, an AI fact-checking assistant for students.
ML verdict: {ml_verdict} ({ml_confidence:.1f}% confidence)
Tone score: {tone_score}/100 | Sentiment: {sentiment["label"]} ({sentiment["compound"]})
Article: \"\"\"{article_text[:800]}\"\"\"
Write 3-4 sentences explaining why this article is likely {ml_verdict}.
Point out 2-3 specific signals. End with one verification tip. English only."""
    return call_gemini(model, prompt, expect_json=False)


def get_multilingual_analysis(model, article_text):
    prompt = f"""
You are VeriFact, a multilingual AI fact-checker for students.
Analyze this article (any language). Translate if needed, then classify.
Do NOT answer Uncertain. Always choose Fake or Real.
Professional journalism with named sources and neutral tone = Real.
Sensational language, unverified claims, emotional manipulation = Fake.
Respond ONLY as JSON (no markdown, no backticks):
{{"verdict":"Fake" or "Real","confidence":0-100,
"explanation":"2-3 sentences in English","signals":["s1","s2","s3"],
"verify_tip":"one actionable tip"}}
Article: \"\"\"{article_text[:1200]}\"\"\""""
    return call_gemini(model, prompt, expect_json=True)


def get_low_confidence_analysis(model, article_text, ml_verdict, ml_confidence):
    prompt = f"""
You are VeriFact. ML model was uncertain (verdict: {ml_verdict}, confidence: {ml_confidence:.1f}%).
Analyze independently. Do NOT answer Uncertain. Always choose Fake or Real.
Professional journalism with named sources and neutral tone = Real.
Sensational language, unverified claims, emotional manipulation = Fake.
Respond ONLY as JSON (no markdown, no backticks):
{{"verdict":"Fake" or "Real","confidence":0-100,
"explanation":"2-3 sentences","signals":["s1","s2"],
"verify_tip":"one specific tip"}}
Article: \"\"\"{article_text[:1000]}\"\"\""""
    return call_gemini(model, prompt, expect_json=True)
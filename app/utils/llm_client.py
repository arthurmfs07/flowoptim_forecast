# http://127.0.0.1:8000/analyze/?ticker=NVDA

# app/utils/llm_client.py
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import requests
from openai import OpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

# Initialize OpenAI client (if available)
client = None
if OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        client = None

# --- Build LLM prompt ---
def build_prompt(ticker, latest_snapshot, anomaly_stats, top_anomalies=None,
                 historical_rows=None, forecast_rows=None):
    lines = [f"Ticker: {ticker}"]

    # Historical context
    if historical_rows:
        lines.append("\nRecent historical daily data:")
        for row in historical_rows:
            lines.append(json.dumps({k: row[k] for k in ["Date","Close","SMA_5","RSI_14","MACD"] if k in row}))

    # Intraday snapshot
    if latest_snapshot:
        lines.append("\nLatest intraday snapshot (15m before close):")
        lines.append(json.dumps({k: latest_snapshot[k] for k in ["Datetime","close","predicted_close","RSI_14","MACD"] if k in latest_snapshot}))

    # Forecast data
    if forecast_rows:
        lines.append("\nFuture forecasts for coming days:")
        for f in forecast_rows:
            lines.append(json.dumps({k: f[k] for k in ["Date","predictions","0.05","0.5","0.95"] if k in f}))

    # Anomaly info
    lines.append("\nAnomaly stats:")
    lines.append(json.dumps(anomaly_stats))
    if top_anomalies:
        lines.append("Recent detected anomalies:")
        for a in top_anomalies:
            lines.append(json.dumps(a))

    lines.append("\nTask: Write 3–5 concise bullet points comparing historical trend, "
                 "intraday snapshot, and upcoming forecasts. Highlight direction, strength, "
                 "and any anomalies or divergences.")
    return "\n".join(lines)

# --- Ollama summarization ---
def summarize_with_ollama(prompt: str, model: str = "mistral") -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt},
            timeout=60,
            stream=True,
        )
        summary_parts = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line.decode("utf-8"))
                if "response" in chunk:
                    summary_parts.append(chunk["response"])
            except Exception:
                continue
        return "".join(summary_parts).strip()
    except Exception as e:
        raise RuntimeError(f"Ollama failed: {e}")

# --- OpenAI summarization ---
def summarize_with_openai(prompt: str, max_tokens: int = 250) -> str:
    if not client:
        raise RuntimeError("OpenAI client not available")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# --- Hugging Face fallback ---
def summarize_with_huggingface(prompt: str, max_length: int = 250) -> str:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    payload = {"inputs": prompt, "parameters": {"max_length": max_length}}
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
    result = resp.json()
    if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
        return result[0]["summary_text"].strip()
    return f"[HF error] {result}"

# --- Master function ---
def summarize_analysis(
    ticker: str,
    latest_snapshot: Dict[str, Any],
    anomaly_stats: Dict[str, Any],
    top_anomalies: Optional[List[Dict[str, Any]]] = None,
    historical_rows: Optional[List[Dict[str, Any]]] = None,
    forecast_rows: Optional[List[Dict[str, Any]]] = None  
) -> Dict[str, Any]:
    prompt = build_prompt(
        ticker,
        latest_snapshot,
        anomaly_stats,
        top_anomalies,
        historical_rows,
        forecast_rows  
    )
    summary = ""
    engine = "fallback"

    # Ollama first
    try:
        summary = summarize_with_ollama(prompt)
        engine = "ollama"
        print("✅ Summary generated with Ollama.")
    except Exception as e:
        print(f"⚠️ Ollama failed: {e}")
        # OpenAI fallback
        try:
            if client:
                summary = summarize_with_openai(prompt)
                engine = "openai"
        except Exception as e2:
            print(f"⚠️ OpenAI failed, falling back to Hugging Face: {e2}")
            try:
                summary = summarize_with_huggingface(prompt)
                engine = "huggingface"
            except Exception as e3:
                print(f"⚠️ HF failed: {e3}")
                summary = (
                    f"- Anomalies found: {anomaly_stats.get('n_anomalies', 0)} / {anomaly_stats.get('n_rows', '?')}\n"
                    "- Check volume, gaps, or data quality.\n"
                    "- Consider retraining if anomalies persist."
                )

    return {"engine": engine, "summary": summary, "prompt": prompt}


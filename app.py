import os
import pickle
import json
import streamlit as st
try:
    import joblib
except Exception:
    # joblib may not be installed in the runtime; fall back to None and use pickle fallback in load_model
    joblib = None
import pandas as pd
from khmernltk import word_tokenize

# Config
MODEL_PATH = r"khmer_pos_crf_model.pkl"  # Update with your model path

st.set_page_config(page_title="Khmer POS Tagging", layout="wide")

st.markdown("<h1 style='text-align:center;'>Khmer POS Tagging</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Use the sidebar to switch between single-sentence, batch upload, and guidelines.</p>", unsafe_allow_html=True)

# ---- Session state init ----
for k, v in {
    "input_text": "",
    "model": None,
    "model_error": "",
    "output_tokens": [],
    "output_tags": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---- Utilities ----
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        return joblib.load(path)
    except Exception:
        pass
    with open(path, "rb") as f:
        return pickle.load(f)

def tokenize(text):
    return word_tokenize(text)

def word2features(sent_tokens, i):
    w = sent_tokens[i]
    features = {
        "bias": 1.0,
        "word": w,
        "lower": w.lower(),
        "len": len(w),
        "is_digit": w.isdigit(),
    }
    for k in range(1, 4):
        features[f"pref{str(k)}"] = w[:k]
        features[f"suf{str(k)}"] = w[-k:]
    if i > 0:
        features["-1:word"] = sent_tokens[i - 1]
    else:
        features["BOS"] = True
    if i < len(sent_tokens) - 1:
        features["+1:word"] = sent_tokens[i + 1]
    else:
        features["EOS"] = True
    return features

def tokens_to_features(tokens):
    return [word2features(tokens, i) for i in range(len(tokens))]

def predict_tags_from_tokens(model, tokens):
    if len(tokens) == 0:
        return []
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None

    def _as_list(x):
        if _np is not None and isinstance(x, _np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        return None

    # sklearn-like
    try:
        if hasattr(model, "predict"):
            feats = tokens_to_features(tokens)
            try:
                preds = model.predict([feats])
            except Exception:
                preds = model.predict([list(tokens)])
            preds = _as_list(preds)
            if preds:
                if len(preds) == 0:
                    return []
                if isinstance(preds[0], (list, tuple)):
                    return list(preds[0])
                if len(preds) == len(tokens):
                    return list(preds)
    except Exception:
        pass

    # python-crfsuite Tagger
    try:
        if hasattr(model, "tag"):
            preds = model.tag(list(tokens))
            preds = _as_list(preds) or list(preds)
            return list(preds)
    except Exception:
        pass

    # generic callable
    try:
        preds = None
        try:
            preds = model(tokens)
        except Exception:
            if hasattr(model, "predict"):
                preds = model.predict(list(tokens))
        if preds is not None:
            preds = _as_list(preds)
            if preds and len(preds) == len(tokens):
                return list(preds)
    except Exception:
        pass

    raise RuntimeError("Model interface not recognized or prediction failed.")

# ---- Load model once ----
if st.session_state["model"] is None and st.session_state["model_error"] == "":
    try:
        st.session_state["model"] = load_model(MODEL_PATH)
    except Exception as e:
        st.session_state["model_error"] = str(e)
LOGO_PATH = r"CADT_logo.png"
if os.path.exists(LOGO_PATH):
    try:
        st.sidebar.image(LOGO_PATH, width=180)  # adjust width as needed
    except Exception:
        # fallback: read bytes if direct path fails in some envs
        try:
            with open(LOGO_PATH, "rb") as f:
                st.sidebar.image(f.read(), width=180)
        except Exception:
            pass
        except Exception:
            pass
        
# 7. Navigation dropdown
menu = st.sidebar.selectbox(
    "Menu",
    ["Sentence Input", 
     "Batch Upload (JSON)", 
     "Guidelines & POS Summary"]
)

# 2. Optional small spacing
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# 3. Project title (bold and centered)
st.sidebar.markdown("## NLP POS Tagging Tool")

# 4. Separator
st.sidebar.markdown("---")

# 5. Project information / description
st.sidebar.markdown("""


This tool supports both single-sentence tagging and batch processing (JSON format) using a custom-trained Khmer POS model.
""")

# 6. Another separator before the menu
st.sidebar.markdown("""
**Project:** Khmer Part-of-Speech Tagging System  
**Author:** Mean Piseth, Sovan Chandara, Oung Chunheng, San Haksou  
**Institution:** CADT (Cambodia Academy of Digital Technology)  
**Year:** 2025  
""")

# Optional: add a footer with version or contact
st.sidebar.markdown("---")
st.sidebar.caption("Version 1.0")


# --- Helper: extract sentences from uploaded json ---
def extract_sentences_from_json(data):
    # Accept:
    # - list of strings: ["សួស្ដី...", ...]
    # - list of dicts: [{"sentence": "..."} , ...]
    # - dict with key 'sentences' -> list
    out = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # common keys
                for k in ("sentence", "text", "khmer"):
                    if k in item and isinstance(item[k], str):
                        out.append(item[k])
                        break
    elif isinstance(data, dict):
        # try common patterns
        if "sentences" in data and isinstance(data["sentences"], list):
            out.extend(extract_sentences_from_json(data["sentences"]))
        else:
            # maybe mapping id->sentence or single sentence
            for k, v in data.items():
                if isinstance(v, str):
                    out.append(v)
    return out

# ---- Views ----
if menu == "Sentence Input":
    # Two-column layout for single sentence tagging (same as original)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input (Khmer sentence)")
        txt = st.text_area(
            "Enter Khmer sentence here",
            value=st.session_state["input_text"],
            height=250,
            placeholder="វាយឃ្លាខ្មែរ​ទៅ​នេះ...",
            key="input_area",
        )
        generate = st.button("Generate")
        st.session_state["input_text"] = txt

    with col2:
        st.subheader("Tagged Output")
        if st.session_state["output_tokens"]:
            df = pd.DataFrame({"Word": st.session_state["output_tokens"], "POS Tag": st.session_state["output_tags"]})
            st.table(df)
        else:
            st.info("No tagged output yet. Enter a Khmer sentence on the left and click Generate.")

    if generate:
        text = st.session_state.get("input_text", "").strip()
        if not text:
            st.warning("Enter a Khmer sentence.")
        elif st.session_state["model_error"]:
            st.error(f"Model load error: {st.session_state['model_error']}")
        else:
            model = st.session_state.get("model")
            if model is None:
                st.error("Model not loaded. Please check the model path and restart the app.")
            else:
                try:
                    tokens = tokenize(text)
                    tags = predict_tags_from_tokens(model, tokens)
                    st.session_state["output_tokens"] = tokens
                    st.session_state["output_tags"] = tags
                except Exception as e:
                    st.error(f"Failed to tag sentence: {e}")

elif menu == "Batch Upload (JSON)":
    st.subheader("Batch predict from JSON")
    st.markdown("Upload a JSON file. Supported formats: list of strings, list of dicts with key 'sentence' or 'text', or {'sentences': [...]}.")

    uploaded = st.file_uploader("Upload JSON file", type=["json"])
    process = st.button("Process File")

    if uploaded is not None:
        try:
            raw = uploaded.read()
            data = json.loads(raw.decode("utf-8"))
            sentences = extract_sentences_from_json(data)
            if not sentences:
                st.warning("No sentences found in uploaded JSON. Check format.")
            else:
                st.info(f"Found {len(sentences)} sentence(s) in uploaded file.")
                st.write("Preview (first 5):")
                for s in sentences[:5]:
                    st.write("-", s)
                st.session_state["batch_sentences"] = sentences
        except Exception as e:
            st.error(f"Failed to read JSON: {e}")

    if process:
        sentences = st.session_state.get("batch_sentences", [])
        if not sentences:
            st.warning("No sentences to process. Upload a file first.")
        elif st.session_state["model_error"]:
            st.error(f"Model load error: {st.session_state['model_error']}")
        else:
            model = st.session_state.get("model")
            if model is None:
                st.error("Model not loaded. Please check the model path and restart the app.")
            else:
                results = []
                failed = 0
                for s in sentences:
                    try:
                        toks = tokenize(s)
                        tags = predict_tags_from_tokens(model, toks)
                        results.append({"sentence": s, "tokens": toks, "tags": tags})
                    except Exception:
                        failed += 1
                        results.append({"sentence": s, "tokens": [], "tags": [], "error": "prediction failed"})
                st.success(f"Processed {len(sentences)} sentences ({failed} failed).")
                # show small table
                preview = [{"sentence": r["sentence"], "tokens": r["tokens"][:6], "tags": r["tags"][:6]} for r in results[:20]]
                st.write("Preview results (first 20):")
                st.json(preview)

                # prepare download
                out_bytes = json.dumps(results, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button(
                    label="Download results as JSON",
                    data=out_bytes,
                    file_name="khmer_pos_results.json",
                    mime="application/json",
                )

elif menu == "Guidelines & POS Summary":
    st.subheader("User Guidelines")
    st.markdown(
        "- Use the Sentence Input for quick one-off tagging.\n"
        "- Use Batch Upload to process many sentences and download full results.\n"
        "- Input JSON should contain Khmer sentences (see upload hint).\n"
        "- Tokenization uses khmernltk; POS tags depend on the model.\n"
    )

    st.subheader("24 POS Tag Summary")
    tags_list = [
        "NOUN", "VERB", "ADJ", "ADV", "PRON or PN", "NUM", "DET", "CONJ",
        "ADP", "PROPN", "PART", "INTJ", "AUX", "PUNCT", "SYM", "X",
        "CLF", "POST", "NEG", "DEM", "POS", "ORD", "QUANT", "COP"
    ]
    tag_desc = [
        "Common noun", "Verb", "Adjective", "Adverb", "Pronoun", "Numeral", "Determiner", "Conjunction",
        "Adposition (pre/postposition)", "Proper noun", "Particle", "Interjection", "Auxiliary verb", "Punctuation", "Symbol", "Other/unknown",
        "Classifier", "Postposition", "Negation", "Demonstrative", "Possessive", "Ordinal", "Quantifier", "Copula"
    ]
    df_tags = pd.DataFrame({"Tag": tags_list, "Description": tag_desc})
    st.table(df_tags)

# Footer with model path
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:gray;'>© Khmer POS Tagging •  Contact: {piseth.mean, chandara.sovan, chhunheng.oung, haksou.sang}@cadt.edu.kh </div>",
    unsafe_allow_html=True,
)







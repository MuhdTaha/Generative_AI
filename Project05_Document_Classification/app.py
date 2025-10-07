import os
import tempfile
import streamlit as st
from classify_ai import DOCUMENT_TYPES
from hybrid_model import hybrid_classify

# ----------------------
# Page Configuration
# ----------------------
st.set_page_config(
    page_title="📁 Intelligent Document Classifier",
    page_icon="🤖",
    layout="wide"
)

# ----------------------
# Title & Description
# ----------------------
st.title("🤖 Intelligent Document Classification (AI + ML Hybrid)")
st.markdown("""
Welcome to the **Hybrid Document Classifier** — an advanced AI app that combines **Gemini reasoning**
with a **TF-IDF + Logistic Regression model** trained on pseudo-labeled data.

**How it works:**
1. Upload a document (`.pdf`, `.txt`, `.docx`).
2. The AI and ML models independently classify it.
3. The app then ensembles both predictions to produce a **final hybrid result**.
""")
st.divider()

# ----------------------
# Session State
# ----------------------
if "hybrid_result" not in st.session_state:
    st.session_state.hybrid_result = None

# ----------------------
# File Uploader
# ----------------------
uploaded_file = st.file_uploader(
    "📤 Upload a document to classify",
    type=["pdf", "txt", "docx"],
    help="Upload a PDF, text, or Word document for classification."
)

# ----------------------
# Classification Trigger
# ----------------------
if uploaded_file:
    st.info(f"📄 File uploaded: **{uploaded_file.name}**")

    if st.button("🔍 Classify Document", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        with st.spinner("🧠 Running hybrid classification... Please wait."):
            try:
                result = hybrid_classify(file_path)
                st.session_state.hybrid_result = result
            except Exception as e:
                st.error(f"❌ Error: {e}")
            finally:
                os.remove(file_path)

# ----------------------
# Display Results
# ----------------------
if st.session_state.hybrid_result:
    result = st.session_state.hybrid_result
    st.divider()
    st.subheader("📊 Classification Results")

    final_label = result.get("final_label", "Unknown")
    final_conf = result.get("final_confidence", 0)
    ai_pred = result.get("ai_prediction", {})
    ml_pred = result.get("ml_prediction", {})

    # --- Hybrid Summary ---
    st.markdown("### 🧩 Final Hybrid Prediction")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Type", final_label)
    with col2:
        st.metric("Confidence", f"{final_conf*100:.1f}%")
    st.success("✅ Hybrid model combines Gemini’s reasoning with statistical ML confidence for best accuracy.")

    st.divider()

    # --- AI Model Output ---
    st.markdown("### 🤖 Gemini AI Prediction")
    st.write(f"**Label:** {ai_pred.get('predicted_class', 'N/A')}")
    st.write(f"**Confidence:** {ai_pred.get('confidence_score', 0):.2f}")
    st.caption(f"_Explanation:_ {ai_pred.get('explanation', 'No explanation available.')}")
    st.divider()

    # --- ML Model Output ---
    st.markdown("### 📈 Classical ML Model (TF-IDF + Logistic Regression)")
    st.write(f"**Label:** {ml_pred.get('label', 'N/A')}")
    st.write(f"**Confidence:** {ml_pred.get('confidence', 0):.2f}")

    # --- Comparison Chart ---
    st.divider()
    st.markdown("### 🔬 Comparison Summary")
    st.bar_chart({
        "Gemini AI Confidence": [ai_pred.get("confidence_score", 0)],
        "ML Confidence": [ml_pred.get("confidence", 0)],
        "Hybrid Confidence": [final_conf],
    })

st.markdown("---")
st.caption("Built by Muhammad Taha using Streamlit + LangChain + Gemini + scikit-learn.")
# 
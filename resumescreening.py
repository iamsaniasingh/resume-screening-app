import streamlit as st
import pickle
import docx
import PyPDF2
import re

# -----------------------------
# Load saved models & vectorizer
# -----------------------------
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# -----------------------------
# Resume Cleaning
# -----------------------------
def cleanResume(txt):
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+", " ", txt)
    txt = re.sub(r"\S+@\S+", " ", txt)
    txt = re.sub(r"\d+", " ", txt)
    txt = re.sub(r"[^a-z\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# -----------------------------
# File Extraction Functions
# -----------------------------
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            p = page.extract_text()
            if p:
                text += p + " "
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = ''
        for p in doc.paragraphs:
            text += p.text + " "
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except:
        try:
            text = file.read().decode('latin-1')
        except Exception as e:
            st.error(f"Error reading TXT: {e}")
            return ""
    return text

# -----------------------------
# File Upload Handler
# -----------------------------
def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == "docx":
        return extract_text_from_docx(uploaded_file)
    elif ext == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        st.warning("Unsupported file type!")
        return ""

# -----------------------------
# Prediction Function
# -----------------------------
def pred(text):
    cleaned = cleanResume(text)

    word_list = cleaned.split()
    if len(word_list) < 20:
        return None

    vec = tfidf.transform([cleaned]).toarray()
    if vec.sum() == 0:
        return None

    prediction = svc_model.predict(vec)
    label = le.inverse_transform(prediction)[0]
    return label

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(
    page_title="🔥 Resume Analyser",
    page_icon="📝",
    layout="centered"
)

# Custom CSS for look
st.markdown("""
    <style>
    .big-title {
        font-size: 2.8rem;
        font-weight: bold;
        color: #4B2E83;
        text-align: center;
    }
    .subtitle {
        font-size: 1.4rem;
        color: #333333;
        text-align: center;
    }
    .upload-box {
        border: 2px dashed #4B2E83;
        padding: 25px;
        border-radius: 12px;
        margin-top: 15px;
    }
    .result-box {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📝 Resume Analyser Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your resume — get smart classification results instantly 🚀</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-box">', unsafe_allow_html=True)
file = st.file_uploader(
    "📂 Upload Resume (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    key="resume_upload"
)
st.markdown('</div>', unsafe_allow_html=True)

if file:
    raw_text = handle_file_upload(file)

    if not raw_text.strip():
        st.warning("⚠️ Couldn't extract meaningful text! Try another file.")
    else:
        st.success("✔️ Text extracted successfully!")

        if st.checkbox("🔍 Show Extracted Text", key="show_text_checkbox"):
            st.text_area("📄 Extracted Resume Content", raw_text, height=300, key="resume_text_area")

        if st.button("💡 Run Prediction", key="run_prediction_button"):
            category = pred(raw_text)
            if category:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.success(f"🎯 **Predicted Category:** {category}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Not enough valid resume content to classify!")
else:
    st.info("📌 Please upload a resume to start analysing.")




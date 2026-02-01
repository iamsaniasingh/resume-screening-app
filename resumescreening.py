import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load saved models & vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Clean resume text
def cleanResume(txt):
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+", " ", txt)
    txt = re.sub(r"\S+@\S+", " ", txt)
    txt = re.sub(r"\d+", " ", txt)
    txt = re.sub(r"[^a-z\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        p = page.extract_text()
        if p:
            text += p + " "
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for p in doc.paragraphs:
        text += p.text + " "
    return text

# Extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except:
        text = file.read().decode('latin-1')
    return text

# File upload handling
def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif ext == "docx":
        return extract_text_from_docx(uploaded_file)
    elif ext == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        return ""

# Predict category
def pred(text):
    cleaned = cleanResume(text)

    # Guard against empty or too-short resumes
    word_list = cleaned.split()
    if len(word_list) < 20:
        return None

    vec = tfidf.transform([cleaned]).toarray() 
    if vec.sum() == 0:   
        return None

    prediction = svc_model.predict(vec)
    label = le.inverse_transform(prediction)[0]
    return label

# Streamlit app UI
st.title("Resume Analyser📝")

file = st.file_uploader("Upload a Resume (PDF, DOCX, TXT)")

if file:
    raw_text = handle_file_upload(file)
    if not raw_text.strip():
        st.warning("⚠️ Couldn't extract meaningful text from uploaded file!")
    else:
        if st.checkbox("Show Extracted Text"):
            st.text_area("Extracted Resume Content", raw_text, height=300)

        if st.button("Run Prediction"):
            category = pred(raw_text)
            if category:
                st.success(f"🔍 Predicted Category: {category}")
            else:
                st.error("❌ Not enough real resume content to classify! Please upload a longer or valid resume.")

 🧠 What This Project Actually Does

✔ Loads a resume dataset  
✔ Preprocesses text (cleaning + TF-IDF)  
✔ Trains / loads a classification model  
✔ Predicts the **field/domain** of an input resume  
✔ Outputs most likely career category  
So if you feed it a resume, it might say:  
> **“Field: Data Science”**  
> **“Field: Full Stack Development”**


## 🗂 Repo Contents

| File / Folder | Purpose |
|---------------|---------|
| `resume_screening.ipynb` | Jupyter notebook with code logic |
| `Resume Screening.csv` | Sample resume dataset |
| `tfidf.pkl` | Saved TF-IDF text vectorizer |
| `encoder.pkl` | Saved label encoder |
| `clf.pkl` | **Trained classification model** (the main brain) |
| `requirements.txt` | Python packages list |
| `README.md` | This file |

🛠 Tech Stack (Expanded & Real)

Core ML & Python

🐍 Python — main language for data + ML

🧠 scikit-learn — TF-IDF, label encoder, classifier

📓 Google COllab - for model training 

🗃 pickle (clf.pkl / tfidf.pkl / encoder.pkl) — saving models

Text Processing & NLP

🔡 TF-IDF Vectorizer — turning resume text into numbers

🧹 Regex / cleaning scripts — text cleanup

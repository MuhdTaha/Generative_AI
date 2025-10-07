# Project 05: Intelligent Document Classification (AI + ML Hybrid)

This project demonstrates an advanced document classification system that combines the power of a large language model (Google's Gemini) with a classical machine learning model (scikit-learn's Logistic Regression). The result is a highly accurate, efficient, and cost-effective hybrid classifier, all wrapped in an interactive Streamlit web application.

The core innovation is using the LLM not just for direct classification, but to bootstrap the creation of a labeled dataset, which is then used to train a faster, more traditional ML model.

---

## 🚀 Key Features

- **Hybrid Intelligence:** Leverages Gemini's deep reasoning to create a high-quality labeled dataset and combines its predictions with a fast scikit-learn model for the final classification.
- **Automated Dataset Creation:** Automatically generates a labeled CSV file from a directory of raw, unlabeled documents (.pdf, .txt, .docx).
- **Efficient & Scalable:** After the initial training, the lightweight scikit-learn model can handle many classifications quickly and at a lower cost than relying solely on LLM calls.
- **Interactive Web App:** A clean and modern Streamlit interface allows users to upload documents and see the classification results from the AI, the ML model, and the final hybrid prediction.
- **Comparative Analysis:** The UI provides a side-by-side comparison of the confidence scores from both models, offering insights into their respective strengths.

---

## ⚙️ How It Works

The project follows a two-stage process: **Training** and **Inference**.

### **1. Data Preparation**
Place your collection of unlabeled documents into the `data/docs/` directory.

### **2. Pseudo-Labeling (AI)**
The `train_model.py` script iterates through your documents. For each one, it calls the Gemini API to get a “pseudo-label,” confidence score, and explanation.  

The results are saved to `data/labeled.csv`.

### **3. Model Training (ML)**
The script then uses this newly created `labeled.csv` dataset to train a TF-IDF Vectorizer and a Logistic Regression classifier.  

The trained model pipeline is saved as `models/tfidf_logreg.pkl`.

### **4. Hybrid Inference**
The `app.py` (Streamlit app) is launched. 

When a user uploads a new document:

- The document is sent to Gemini for an AI-powered prediction.
- The document is simultaneously sent to the saved scikit-learn model for a fast, statistical prediction.
- A weighted algorithm combines both outputs to produce a final, highly reliable classification.

---

## 🛠️ Tech Stack

- **Backend:** Python  
- **Web Framework:** Streamlit  
- **AI Orchestration:** LangChain  
- **Language Model:** Google Gemini (gemini-2.5-flash)  
- **Machine Learning:** Scikit-learn (TF-IDF, Logistic Regression)  
- **Data Handling:** Pandas, NumPy  
- **Model Persistence:** Joblib  

---

## 📂 Project Structure

```
.
├── 📄 app.py               # Streamlit web application
├── 📄 classify_ai.py        # Logic for classifying documents using Gemini API
├── 📄 hybrid_model.py      # Training and hybrid prediction logic
├── 📄 train_model.py       # Script for pseudo-labeling and model training
├── 📁 data/
│   ├── 📁 docs/             # <-- Place your raw documents here
│   └── 📄 labeled.csv      # (Generated automatically)
├── 📁 models/
│   └── 📄 tfidf_logreg.pkl  # (Generated automatically)
├── 📄 requirements.txt     # Dependencies
└── 📄 .env                 # Stores your API key
```

---

## 📋 Setup and Installation

### **1. Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### **2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **3. Create a `.env` file**
Add your Google API key:
```bash
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
```

---

## ▶️ How to Run the Application

### **Step 1: Prepare Your Data**
Place your sample documents (`.pdf`, `.txt`, `.docx`) inside the `data/docs/` directory.

### **Step 2: Train the Hybrid Model**
Run:
```bash
python train_model.py
```
This will use Gemini to label your data and train the ML model.

### **Step 3: Launch the Streamlit Web App**
```bash
streamlit run app.py
```
A browser tab will open at **http://localhost:8501**, where you can upload documents and see hybrid classifications in real time.

---

## 🧠 Notes

- The more diverse your dataset, the better the model performance.
- Ensure your `.env` file is correctly configured before running the app.
- You can retrain the model anytime by re-running `train_model.py`.

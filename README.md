
# NeethiAI - Your Legal AI Companion

**NeethiAI** is a Python-based generative AI legal assistant that empowers citizens to better understand and act on legal matters. It combines Gemini Pro (by Google), OCR, web scraping, and search capabilities to provide intelligent summaries, legal advice, document analysis, and lawyer lookup in real-time through an interactive Streamlit interface.

---


##  App Modules

| Feature                        | Description |
|-------------------------------|-------------|
| **Summarize Legal Documents** | Upload a PDF or Word document and receive a legal summary + Q&A |
| **Legal Notice Generator**    | Generate formal legal notices for disputes or complaints |
| **Direct Legal Q&A**          | Ask a question and get an answer based on Indian laws |
| **Tax Advisory**              | Get structured, sectioned tax advice with latest updates |
| **Fake Notice Detector**      | Upload legal notices/images to check for fraud using OCR |
| **Lawyer Finder**             | Search LawRato for lawyers by city and specialization |

---

##  Installation

###  Prerequisites

- Python 3.8 or higher
- Google Gemini API Key ([Get it here](https://aistudio.google.com/apikey))

###  Install Dependencies

```bash
pip install -r requirements.txt
```
pip install streamlit PyPDF2 python-docx easyocr opencv-python-headless duckduckgo-search google-generativeai beautifulsoup4 requests
---


##  Usage

### 1. Run the Streamlit App

```bash
streamlit run app.py
```

### 2. Enter Your Gemini API Key

The app requires a Gemini API key for AI-powered legal understanding.

---

##  File Structure

```
neethiAI/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Dependency list
├── README.md                 # Project description
├── nAI_logo.png              # Logo image for UI
└── modules/                  # (Optional) Future modular refactor
```

---

##  Dependencies

- **Streamlit**: Web UI framework
- **Gemini Pro (google.generativeai)**: Core AI model for reasoning and content generation
- **PyPDF2 / python-docx**: Document parsing
- **EasyOCR + OpenCV**: Image-based notice scanning
- **DuckDuckGo Search + BeautifulSoup**: Legal content scraping
- **Requests**: Web requests for scraping and APIs

---

##  Future Improvements

- Add user login and session history
- Add support for multilingual legal queries
- Integrate live chat with real lawyers
- Offer downloadable legal reports (PDF format)
- Build mobile app version (Flutter or PWA)

---

##  License

This project is open-source and available under the **MIT License**.

---

##  Contact

Created by [Hariharan G.](mailto:hariharangl005@gmail.com) – AI & Data Science Engineering student

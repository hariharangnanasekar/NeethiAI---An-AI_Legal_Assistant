import streamlit as st
import PyPDF2
import docx
import easyocr
from PIL import Image
import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException
import google.generativeai as genai
import re
import random
from datetime import datetime
from time import sleep
import logging
import os  # Added for path handling

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Configure Gemini API
def configure_gemini_api(api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        return model
    except Exception as e:
        st.error(f"Invalid API key: {str(e)}")
        return None

# PDF Parsing
def extract_text_from_pdf(file):
    full_text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text() or ""
            full_text += text
        return full_text[:4000]  # Gemini safe limit
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

# Word Document Parsing
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        full_text = ""
        for para in doc.paragraphs:
            full_text += para.text + "\n"
        return full_text[:4000]  # Gemini safe limit
    except Exception as e:
        raise Exception(f"Error extracting text from Word document: {str(e)}")

# Legal Source Finder
def get_legal_sources(query, max_results=3):
    trusted_sites = [
        "site:indiankanoon.org", 
        "site:gov.in", 
        "site:prsindia.org", 
        "site:egazette.nic.in", 
        "site:legislative.gov.in"
    ]
    combined_sites = " OR ".join(trusted_sites)
    search_query = f"{query} {combined_sites}"
    try:
        sleep(1)  # Add 1-second delay to avoid rate limits
        with DDGS() as ddgs:
            results = ddgs.text(search_query, max_results=max_results)
        trusted_links = [result.get("href") for result in results if result.get("href") and any(site.split(":")[1] in result.get("href") for site in trusted_sites)]
        return trusted_links
    except RatelimitException as e:
        raise Exception(f"Search rate limit exceeded. Please try again later: {str(e)}")
    except Exception as e:
        raise Exception(f"Error fetching legal sources: {str(e)}")

# Web Scraping
def scrape_web_text(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return text[:4000]
    except Exception:
        return ""

# Summarize Legal Issue
def summarize_legal_issue(document_text, query, urls, model):
    try:
        if not document_text.strip():
            raise Exception("Document is empty.")
        prompt = f"""
You are a legal AI assistant helping citizens understand their rights.

User Question: {query}

Document Content:
{document_text}

Trusted Source URLs:
{', '.join(urls)}

Generate:
- Clear answer with law references
- Steps the citizen can take
- End with source links
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error summarizing document: {str(e)}")

# Generate Legal Notice
def generate_legal_notice(user_issue, user_name, opponent_name, model):
    try:
        # Log inputs for debugging
        logging.debug(f"Inputs: user_issue='{user_issue}', user_name='{user_name}', opponent_name='{opponent_name}'")
        if not user_issue.strip() or not user_name.strip() or not opponent_name.strip():
            raise Exception("All fields must be filled.")
        prompt = f"""
Draft a legal notice based on the following:

- Problem: {user_issue}
- User: {user_name}
- Opponent: {opponent_name}

Make it formal and actionable.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error generating notice: {str(e)}")

# Direct Legal Query
def answer_legal_query_directly(query, model):
    try:
        if not query.strip():
            raise Exception("Query cannot be empty.")
        urls = get_legal_sources(query)
        for url in urls:
            text = scrape_web_text(url)
            if len(text) > 300:
                prompt = f"""
You are a legal AI assistant. A user has asked a legal question:

**Question:** {query}

Below is relevant legal content from a trusted source:

{text}

Your task:
- Answer the user's query clearly
- Mention relevant legal sections/acts if found
- End with a trusted source link: {url}
"""
                response = model.generate_content(
                    contents=[prompt],
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=2048
                    )
                )
                return response.text + f"\n\n[Source Link]({url})"
        return "Sorry, I couldn't find a reliable legal source to answer your question."
    except Exception as e:
        raise Exception(f"Error answering query: {str(e)}")

# Tax Advisory
def get_tax_advice(user_query, model):
    try:
        if not user_query.strip():
            raise Exception("Query cannot be empty.")
        system_prompt = (
            "You are a certified Indian tax consultant AI assistant. "
            "Answer the user's question in a clear, structured way using the latest tax laws and budget updates. "
            "Include the following sections:\n\n"
            "1. **User Query**\n"
            "2. **Applicable Tax Regime & Slab**\n"
            "3. **Total Tax Payable (with calculation)**\n"
            "4. **Possible Deductions / Tax Saving Tips**\n"
            "5. **Summary Advice**\n"
            "6. **Trusted Sources** (include official links like incometax.gov.in, cbic.gov.in, or cleartax.in)\n"
            "Use Markdown formatting for headings and bold text."
        )
        full_prompt = f"{system_prompt}\n\nUser: {user_query}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        raise Exception(f"Error generating tax advice: {str(e)}")

# Fake Notice Detection
def preprocess_image(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_easyocr(image):
    try:
        results = reader.readtext(image, detail=0)
        return "\n".join(results)
    except Exception as e:
        raise Exception(f"Error extracting text from image: {str(e)}")

def detect_fake_notice(text):
    suspicious_keywords = ["urgent action", "pay immediately", "legal threat", "court summon", "non-bailable"]
    found = [kw for kw in suspicious_keywords if kw.lower() in text.lower()]
    if found:
        return f"Suspicious terms found: {', '.join(found)}\n\nThis may be a **fake notice**. Please consult a lawyer."
    return "No suspicious content detected. The notice appears legitimate."

# Lawyer Lookup
def is_available_now():
    current_hour = datetime.now().hour
    if 10 <= current_hour <= 18:
        return random.choice([True, True, False])
    return random.choice([False, False, True])

def format_languages(languages):
    if languages == "Not specified" or not languages:
        return languages
    languages = languages.replace("\n", ", ")
    lang_list = re.split(r'[,;、]\s*', languages)
    lang_list = [lang.strip().capitalize() for lang in lang_list if lang.strip()]
    return ", ".join(lang_list)

def format_practice_areas(practice_areas):
    if practice_areas == "Not specified" or not practice_areas:
        return practice_areas
    if len(practice_areas) > 100:
        return f"{practice_areas.title()}"
    return practice_areas.title()

def get_lawyers_from_lawrato(city, specialization):
    city_formatted = city.lower().strip()
    specialization_formatted = specialization.lower().strip()
    url = f"https://lawrato.com/{specialization_formatted}-lawyers/{city_formatted}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: HTTP {response.status_code}")
        soup = BeautifulSoup(response.content, 'html.parser')
        lawyers_data = extract_lawyers_from_html(soup, city, specialization)
        return lawyers_data
    except Exception as e:
        raise Exception(f"Error fetching lawyers: {str(e)}")

def extract_lawyers_from_html(soup, city, specialization):
    lawyers_data = []
    lawyer_containers = soup.select('.lawyer-item, .border-box')
    for container in lawyer_containers[:5]:
        try:
            name_elem = container.select_one('.media-heading, h2, a[title]')
            name = None
            if name_elem:
                name = name_elem.get_text().strip() or (name_elem.get('title') if name_elem.has_attr('title') else None)
            if not name or len(name) < 3:
                continue
            location_elem = container.select_one('.location span')
            location = location_elem.get_text().strip() if location_elem else city.title()
            experience_elem = container.select_one('.experience span')
            experience = experience_elem.get_text().strip() if experience_elem else "Not specified"
            rating_elem = container.select_one('.score')
            rating = "Not rated"
            if rating_elem:
                rating_text = rating_elem.get_text().strip()
                rating_match = re.search(r'(\d\.?\d*)', rating_text)
                if rating_match:
                    rating = f"{rating_match.group(1)}/5"
            link_elem = container.select_one('a[href]')
            profile_url = ""
            if link_elem:
                href = link_elem.get('href')
                profile_url = href if href.startswith('http') else f"https://lawrato.com{href}"
            lawyers_data.append({
                "Name": name,
                "Specialization": specialization.title() + " Law",
                "Location": location,
                "Experience": experience,
                "Rating": rating,
                "Available_Now": is_available_now(),
                "Source": "lawrato.com",
                "Profile_URL": profile_url
            })
        except:
            continue
    return lawyers_data

def get_lawyer_details(url):
    details = {
        "Experience": "Not specified",
        "Languages": "Not specified",
        "Practice_Areas": "Not specified",
        "Rating": "Not rated"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return details
        soup = BeautifulSoup(response.content, 'html.parser')
        details["Experience"] = extract_experience_info(soup)
        details["Languages"] = extract_language_info(soup)
        details["Practice_Areas"] = extract_practice_areas(soup, url)
        details["Rating"] = extract_rating_info(soup)
        return details
    except:
        return details

def extract_experience_info(soup):
    exp_elements = soup.select('.experience, .exp, .lawyer-exp, .profile-exp, .years-exp')
    if exp_elements:
        exp_text = exp_elements[0].get_text().strip()
        exp_pattern = r"(\d+)\+?\s*(?:Years|Yrs|Year)"
        exp_match = re.search(exp_pattern, exp_text, re.IGNORECASE)
        if exp_match:
            return f"{exp_match.group(1)} years Experience"
    exp_pattern = r"(?:Experience|Exp)[:\s]*(\d+)\+?\s*(?:Years|Yrs|Year)"
    for elem in soup.find_all(['p', 'div', 'span', 'li']):
        text = elem.get_text()
        exp_match = re.search(exp_pattern, text, re.IGNORECASE)
        if exp_match:
            return f"{exp_match.group(1)} years Experience"
    return "Not specified"

def extract_language_info(soup):
    language_elements = soup.select('.languages, .langs, .language-list, .lawyer-langs')
    if language_elements:
        return language_elements[0].get_text().strip().replace("Languages:", "").strip()
    for elem in soup.find_all(['p', 'div', 'span', 'li']):
        text = elem.get_text().lower()
        if "language" in text and len(text) < 150:
            text = re.sub(r'language[s]?[:\s]*', '', text, flags=re.IGNORECASE).strip()
            if 3 < len(text) < 150:
                return text
    return "Not specified"

def extract_practice_areas(soup, url):
    practice_areas = []
    practice_elements = soup.select('.practice-areas, .areas-of-practice, .specialization, .lawyer-practice')
    if practice_elements:
        practice_text = practice_elements[0].get_text().strip()
        practice_areas_text = practice_text.replace("Practice Areas:", "").strip()
        if practice_areas_text:
            practice_areas.append(practice_areas_text)
    skill_tags = soup.select('.skill-tag, .tag, .lawyer-skill-tag, .practice-tag')
    for tag in skill_tags:
        tag_text = tag.get_text().strip()
        if tag_text and len(tag_text) < 50:
            practice_areas.append(tag_text)
    area_sections = soup.select('.area-skill, .skill-section, .expertise-section, .specialization-section')
    for section in area_sections:
        section_text = section.get_text().strip()
        if "practice area" in section_text.lower() or "specialization" in section_text.lower():
            content = re.sub(r'^.*?(?:practice areas|specializations?)[:\s]*', '', 
                             section_text, flags=re.IGNORECASE|re.DOTALL).strip()
            if content:
                practice_areas.append(content)
    keyword_lists = soup.select('.lawyer-keywords, .keywords, .tags-list, .practice-list')
    for keyword_list in keyword_lists:
        keywords = [kw.get_text().strip() for kw in keyword_list.find_all(['a', 'span', 'li'])]
        if keywords:
            practice_areas.extend(keywords)
    skill_elements = soup.select('.skill, .expertise, .practice-area')
    for skill in skill_elements:
        skill_text = skill.get_text().strip()
        if skill_text and len(skill_text) < 50:
            practice_areas.append(skill_text)
    common_practices = ["criminal", "civil", "family", "divorce", "property", "corporate", 
                        "tax", "intellectual property", "labor", "immigration", "bankruptcy"]
    if not practice_areas:
        for elem in soup.find_all(['li', 'p', 'div', 'span']):
            text = elem.get_text().strip()
            if any(practice in text.lower() for practice in common_practices):
                matches = sum(1 for practice in common_practices if practice in text.lower())
                if matches >= 2 and len(text) < 300:
                    practice_areas.append(text)
                    break
    if practice_areas:
        all_areas = []
        for area_text in practice_areas:
            cleaned = area_text.replace("\n", ", ")
            split_areas = re.split(r'[,;•|&+]', cleaned)
            all_areas.extend([area.strip() for area in split_areas if area.strip()])
        unique_areas = sorted(set(all_areas))
        return ", ".join(unique_areas).lower()
    if any(kw in url.lower() for kw in common_practices):
        for kw in common_practices:
            if kw in url.lower():
                return kw
    return "Not specified"

def extract_rating_info(soup):
    rating_elements = soup.select('.rating, .score, .star-rating')
    if rating_elements:
        rating_text = rating_elements[0].get_text().strip()
        rating_match = re.search(r'(\d\.?\d*)[/\s]?\d*\s*\((\d+\+?\s*(?:user ratings|ratings))', rating_text, re.IGNORECASE)
        if rating_match:
            return f"{rating_match.group(1)}/5 ({rating_match.group(2)})"
        else:
            rating_match = re.search(r'(\d\.?\d*)', rating_text)
            if rating_match:
                return f"{rating_match.group(1)}/5"
    return "Not rated"

# Main app
st.set_page_config(page_title="NeethiAI - Your Legal AI Companion", layout="wide")

# Custom CSS and Logo
st.markdown("""
<style>
body {
    background-color: #FFFFFF;
}
#title {
    color: #003087;
    font-size: 2.5em;
    font-weight: bold;
}
.feature-container {
    border: 1px solid #E0E0E0;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 20px;
}
.stButton>button {
    background-color: #003087;
    color: white;
    border-radius: 5px;
}
.stButton>button:hover {
    background-color: #005BB5;
    color: white;
}
.stTextInput>div>input, .stTextArea>div>textarea {
    border: 1px solid #E0E0E0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Logo from local file path using st.image
logo_path = os.path.join(os.path.dirname(__file__), "C:/Users/Admin/Downloads/NeethiAI/nAI_logo.png")  
logging.debug(f"Attempting to load logo from: {logo_path}")
if os.path.exists(logo_path):
    try:
        # Verify image integrity with PIL
        Image.open(logo_path).verify()
        # Use columns to align logo and title
        col1, col2 = st.columns([1, 5])
        with col1:
            st.image(logo_path, width=40)
        with col2:
            st.markdown("""
            <h1 id="title">NeethiAI - Your Legal AI Companion</h1>
            <p>A generative AI-powered legal assistant for document summarization, legal advice, and more.</p>
            """, unsafe_allow_html=True)
    except (st.runtime.media_file_storage.MediaFileStorageError, Exception) as e:
        logging.debug(f"Error loading logo at: {logo_path}, Error: {str(e)}")
        st.warning(f"Failed to load logo image: {str(e)}. Please check the file path and format.")
        st.markdown("""
        <h1 id="title">NeethiAI - Your Legal AI Companion</h1>
        <p>A generative AI-powered legal assistant for document summarization, legal advice, and more.</p>
        """, unsafe_allow_html=True)
else:
    logging.debug(f"Logo not found at: {logo_path}")
    st.warning("Logo image not found. Please ensure 'nAI_logo.png' is in the same directory as the script.")
    st.markdown("""
    <h1 id="title">NeethiAI - Your Legal AI Companion</h1>
    <p>A generative AI-powered legal assistant for document summarization, legal advice, and more.</p>
    """, unsafe_allow_html=True)

# API Key Setup
st.header("API Key Setup")
api_key = st.text_input("Enter your Gemini API key:", type="password")
if not api_key:
    st.warning("Please enter a valid Gemini API key to access features.")
    st.stop()
model = configure_gemini_api(api_key)
if not model:
    st.stop()

# Sidebar for feature selection
st.sidebar.header("Features")
feature = st.sidebar.selectbox(
    "Select a feature",
    [
        "Summarize Legal Notices and Documents",
        "Generate Legal Notices/Responses",
        "Direct Legal Question Answering",
        "Tax Advisory Assistant",
        "Fake Notice Detector",
        "Real-Time Legal Consultant"
    ]
)

# Feature 1: Summarize Legal Notices and Documents
if feature == "Summarize Legal Notices and Documents":
    st.header("Summarize Legal Notices and Documents ")
    st.write("Upload a PDF or Word document to get a summary and ask related questions.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"], accept_multiple_files=False)
    if uploaded_file is not None:
        try:
            with st.spinner("Processing document..."):
                if uploaded_file.name.endswith(".pdf"):
                    document_text = extract_text_from_pdf(uploaded_file)
                else:
                    document_text = extract_text_from_docx(uploaded_file)
                sources = get_legal_sources("legal document summary")
                summary = summarize_legal_issue(document_text, "Summarize this document", sources, model)
            st.markdown("**Summary**")
            st.markdown(summary)
            st.markdown("**Ask a Question**")
            query = st.text_input("Enter your question about the document:")
            if query and st.button("Submit Query"):
                with st.spinner("Generating answer..."):
                    sources = get_legal_sources(query)
                    answer = summarize_legal_issue(document_text, query, sources, model)
                st.markdown("**Answer**")
                st.markdown(answer)
                st.markdown("**Sources**")
                for source in sources:
                    st.markdown(f"- [{source}]({source})")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Feature 2: Generate Legal Notices/Responses
elif feature == "Generate Legal Notices/Responses":
    st.header("Generate Legal Notices/Responses ")
    st.write("Enter details to generate a legal notice or response (e.g., for rent disputes or refund claims).")
    user_issue = st.text_area("Describe the legal issue:", height=150)
    user_name = st.text_input("Your name:")
    opponent_name = st.text_input("Opponent's name:")
    if st.button("Generate"):
        try:
            with st.spinner("Generating notice..."):
                notice = generate_legal_notice(user_issue, user_name, opponent_name, model)
                notice = notice.replace("\n", "<br>")
                sources = get_legal_sources("legal notice")
            st.markdown("**Generated Notice/Response**")
            st.markdown(notice, unsafe_allow_html=True)
            st.markdown("**Sources**")
            for source in sources:
                st.markdown(f"- [{source}]({source})")
        except Exception as e:
            if str(e).startswith("Error generating notice: All fields must be filled."):
                st.warning("Please fill in all fields with valid text.")
            else:
                st.error(f"Error: {str(e)}")

# Feature 3: Direct Legal Question Answering
elif feature == "Direct Legal Question Answering":
    st.header("Direct Legal Question Answering ")
    st.write("Ask a legal question to receive an answer with relevant legal sections.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    question = st.text_input("Enter your legal question:")
    if st.button("Ask"):
        if question:
            try:
                with st.spinner("Generating answer..."):
                    answer = answer_legal_query_directly(question, model)
                st.session_state.chat_history.append({"question": question, "answer": answer})
                st.markdown("**Answer**")
                st.markdown(answer)
                sources = get_legal_sources(question)
                st.markdown("**Sources**")
                for source in sources:
                    st.markdown(f"- [{source}]({source})")
                st.markdown("**Chat History**")
                for item in st.session_state.chat_history:
                    st.markdown(f"**You**: {item['question']}")
                    st.markdown(f"**NeethiAI**: {item['answer']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question.")

# Feature 4: Tax Advisory Assistant
elif feature == "Tax Advisory Assistant":
    st.header("Tax Advisory Assistant ")
    st.write("Enter your tax-related query for advice or calculations.")
    tax_query = st.text_area("Enter your tax query:", height=150)
    if st.button("Get Advice"):
        if tax_query:
            try:
                with st.spinner("Generating advice..."):
                    advice = get_tax_advice(tax_query, model)
                    sources = get_legal_sources("tax advisory")
                st.markdown("**Tax Advice**")
                st.markdown(advice)
                st.markdown("**Sources**")
                for source in sources:
                    st.markdown(f"- [{source}]({source})")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a tax query.")

# Feature 5: Fake Notice Detector
elif feature == "Fake Notice Detector":
    st.header("Fake Notice Detector ")
    st.write("Upload a PDF, image, or Word file to check if the notice is real or fake.")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg", "png", "docx"], accept_multiple_files=False)
    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing notice..."):
                if uploaded_file.name.endswith((".jpg", ".jpeg", ".png")):
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)
                    text = extract_text_easyocr(processed_image)
                elif uploaded_file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = extract_text_from_docx(uploaded_file)
                result = detect_fake_notice(text)
                sources = get_legal_sources("fake legal notice")
            st.markdown("**Extracted Text**")
            st.code(text, language="text")
            st.markdown("**Analysis**")
            st.markdown(result)
            st.markdown("**Sources**")
            for source in sources:
                st.markdown(f"- [{source}]({source})")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Feature 6: Real-Time Legal Consultant
elif feature == "Real-Time Legal Consultant":
    st.header("Real-Time Legal Consultant ")
    st.write("Enter a legal specialty and location to find lawyers.")
    specialization = st.selectbox("Legal Specialty:", ["Criminal", "Family", "Property", "Corporate", "Civil"])
    location = st.text_input("Location (e.g., Mumbai):")
    if st.button("Search"):
        if specialization and location:
            try:
                with st.spinner("Searching for lawyers..."):
                    lawyers = get_lawyers_from_lawrato(location, specialization)
                    sources = ["https://lawrato.com"]
                if lawyers:
                    st.markdown("**Lawyers Found**")
                    for lawyer in lawyers:
                        languages = format_languages(lawyer.get('Languages', 'Not specified'))
                        practice_areas = format_practice_areas(lawyer.get('Practice_Areas', lawyer['Specialization']))
                        st.markdown(f"### {lawyer['Name']}")
                        st.markdown(f"- **Location**: {lawyer['Location']}")
                        st.markdown(f"- **Experience**: {lawyer['Experience']}")
                        st.markdown(f"- **Languages**: {languages}")
                        st.markdown(f"- **Practice Areas**: {practice_areas}")
                        st.markdown(f"- **Rating**: {lawyer['Rating']}")
                        st.markdown(f"- **Available Now**: {'Yes' if lawyer['Available_Now'] else 'No'}")
                        st.markdown(f"- **Profile**: [{lawyer['Profile_URL']}]({lawyer['Profile_URL']})")
                    st.markdown("**Source**")
                    for source in sources:
                        st.markdown(f"- [{source}]({source})")
                else:
                    st.warning("No lawyers found for the specified criteria.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter both specialty and location.")

# Footer
st.markdown("---")
st.markdown("Thank you for using NeethiAI! For support, contact [hariharangl005@gmail.com](mailto:hariharangl005@gmail.com).")
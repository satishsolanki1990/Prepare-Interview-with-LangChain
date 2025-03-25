import os
import re
import io
import json
import streamlit as st
from PIL import Image
import docx
import pandas as pd
import pytesseract
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_path
from nltk.tokenize import word_tokenize
# from textblob import TextBlob
import spacy
from spacy.matcher import PhraseMatcher
import nltk
import openai
import utility
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
load_dotenv()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class ATS_System:
    def __init__(self):
        st.set_page_config(layout="wide")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            st.error("API key not found. Please set it in the environment.")
        else:
            openai.api_key = self.api_key

        self.init_ui()
        self.resume_dirs = {
            "raw": "user_resumes",
            "above": "ATS_SCORE_ABOVE_55",
            "below": "ATS_SCORE_BELOW_55"
        }
        for dir_path in self.resume_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        self.IT_SKILLS = utility.TECHNICAL_SKILLS

        self.input_prompt1 = utility.PROMPT_TEMPLATES['tech_hr_review']

        self.input_prompt3 = utility.PROMPT_TEMPLATES['ats_review_simple']

        self.input_prompt4 = utility.PROMPT_TEMPLATES['ats_review_advanced']


    def init_ui(self):
        st.markdown("## Resume & JD Compatibility Checker")
        st.markdown("Upload your job description and resume to evaluate compatibility and receive insights.")

       
        
        self.output_placeholder = st.empty()
        self.response_placeholder = st.empty()
        if 'preprocess_JD' not in st.session_state:
            st.session_state.preprocess_JD = None
        if 'preprocess_doc_content' not in st.session_state:
            st.session_state.preprocess_doc_content = None
        if 'doc_content' not in st.session_state:
            st.session_state.doc_content = None
        if 'job_description_preprocess' not in st.session_state:
            st.session_state.job_description_preprocess = None
        if 'preprocess_doc_content_4' not in st.session_state:
            st.session_state.preprocess_doc_content_4 = None
        
        self.jd_skills_normalized = None
        self.resume_skills_normalized = None
        self.matching_skills = None
        self.similarity_score_1 = 0
        self.similarity_score = 0


    def extract_text_from_pdf(self, uploaded_file):
        try:
            temp_file = os.path.join(self.resume_dirs['raw'], uploaded_file.name)
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            images = convert_from_path(temp_file)
            os.remove(temp_file)
            return "".join([pytesseract.image_to_string(img) for img in images])
        except Exception as e:
            st.error(f"PDF processing error: {e}")
            return None

    def extract_text_from_docx(self, uploaded_file):
        try:
            doc = docx.Document(uploaded_file)
            return '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception as e:
            st.error(f"DOCX processing error: {e}")
            return None
    
    def preprocess_text(self, text):
        text = text.replace(" @", "@").replace("@ ", "@")
        custom_punctuation = list(punctuation)
        custom_punctuation.remove('@')
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        return ' '.join([t for t in tokens if t not in stop_words and t not in custom_punctuation]).replace(" @ ", "@").replace("'", '')

    
    def save_resume_file(self, uploaded_file, category="raw"):
        try:
            target_path = os.path.join(self.resume_dirs[category], uploaded_file.name)
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return target_path
        except Exception as e:
            st.error(f"File save error: {e}")
            return None
    
    def ATS_RESUME_ABOVE_55(self, uploaded_file):
        try:
            os.makedirs(self.RESUME_FOLDER_1, exist_ok=True)
            file_path = os.path.join(self.RESUME_FOLDER_1, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path
        except Exception as e:
            st.error(f"Error saving the file: {e}")
            return None

    def ATS_RESUME_BELOW_55(self, uploaded_file):
        try:
            os.makedirs(self.RESUME_FOLDER_2, exist_ok=True)
            file_path = os.path.join(self.RESUME_FOLDER_2, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path
        except Exception as e:
            st.error(f"Error saving the file: {e}")
            return None
    
    def extract_skills(self, text, skills):
        nlp = spacy.blank("en")
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        matcher.add("SKILLS", [nlp.make_doc(skill) for skill in skills])
        doc = nlp(text)
        return {doc[start:end].text for _, start, end in matcher(doc)}

    def calculate_cosine_similarity(self, jd_terms, matched_terms):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([
            self.preprocess_text(', '.join(jd_terms)),
            self.preprocess_text(', '.join(matched_terms))
        ])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def normalize_skill_terms(self, skill_terms):
        return {term.strip().lower() for term in skill_terms}

    def rate_similarity(self, score):
        if score >= 0.60:
            return ("Strong Match", "#4CAF50")
        elif score >= 0.50:
            return ("Moderate Match", "#FFC107")
        else:
            return ("Weak Match", "#F44336")

    def display_skills_and_similarity(self, jd_text, resume_text):
        jd_skills = self.extract_skills(jd_text, self.IT_SKILLS)
        resume_skills = self.extract_skills(resume_text, self.IT_SKILLS)
        self.jd_skills_normalized = self.normalize_skill_terms(jd_skills)
        self.resume_skills_normalized = self.normalize_skill_terms(resume_skills)
        self.matching_skills = self.jd_skills_normalized.intersection(self.resume_skills_normalized)
        if not self.matching_skills:
            st.markdown(f"No matching skills found between JD and Resume.", unsafe_allow_html=True)
            self.similarity_score = 0
        else:
            self.similarity_score = self.calculate_cosine_similarity(self.jd_skills_normalized, self.matching_skills)
            suggestion, color = self.rate_similarity(self.similarity_score)
            st.markdown(f"{suggestion}", unsafe_allow_html=True)
        data = {
            "Job Description Skills": [', '.join(self.jd_skills_normalized)],
            "Resume Skills": [', '.join(self.resume_skills_normalized)],
            "Matching Skills between Job Description & Resume": [', '.join(self.matching_skills) if self.matching_skills else 'None'],
            "Cosine Similarity": self.similarity_score * 100
            }
        df = pd.DataFrame(data)
        st.markdown("""
            """, unsafe_allow_html=True)
        # st.markdown(f"Cosine Similarity between Job Description Skills and Matching Resume Skills as per JD:- {self.similarity_score * 100:.2f}%", unsafe_allow_html=True)
        st.markdown(df.to_html(classes='dataframe', index=False), unsafe_allow_html=True)

    def extract_jd_info(self, job_description):
        input_prompt_jd_extraction = utility.prompts['job_description_extraction']
        response = self.get_openai_response(None,job_description, input_prompt_jd_extraction)
        if response:
            try:
                jd_data = json.loads(response)
                return jd_data
            except json.JSONDecodeError:
                st.error("Error: Gemini response is not in JSON format.")
                return None
        else:
            return None

    def get_openai_response(self, input_text, doc_content, prompt):
        try:
            if not input_text:
                st.error("Job description is missing. Please enter a job description.")
                return None
            if not doc_content:
                st.error("Resume content is missing. Please upload a resume.")
                return None
            if not prompt:
                st.error("Prompt is missing. Please provide a valid prompt.")
                return None
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": f"Job Description: {input_text}\n Resume:{doc_content}\n {prompt}"},
                ]
            )
            return response.choices.message['content']
        except Exception as e:
            st.error(f"Error during API call: {e}")
            return None
    
    def generate_cover_letter(self, job_description, resume_content):
        cover_letter_agent = Agent(
            role="Cover Letter Writer",
            goal="Craft a concise, professional, and tailored cover letter.",
            backstory=(
                "You specialize in writing compelling cover letters tailored "
                "to job descriptions and resumes. Your goal is to keep it concise, focusing on key strengths and qualifications."
            ),
            memory=False,
            tools=[]
        )
        cover_letter_task = Task(
            description=(
                "Generate a concise and impactful cover letter based on the provided resume and job description."
                "The cover letter should be professional, personalized, and highlight the candidate's skills strengths"
                "\nInputs:\n- Resume: {resume}\n- Job Description: {job_description}"
            ),
            expected_output="A structured, concise cover letter with an Introduction, Key Strengths, and Closing.",
            agent=cover_letter_agent
        )
        cover_letter_crew = Crew(
            agents=[cover_letter_agent],
            tasks=[cover_letter_task],
            process=Process.sequential
        )
        inputs = {
            'resume': resume_content,
            'job_description': job_description
        }
        try:
            result = cover_letter_crew.kickoff(inputs=inputs)
            task_output = cover_letter_task.output
            return task_output.raw
        except Exception as e:
             st.error(f"Error during cover letter generation: {str(e)}")
    
    def ask_openai(self, question, resume_text):
         prompt = f"""
            You are an expert resume analyzer. Answer the following question based only on the provided resume content. Do not include any external knowledge.
            Resume:
            {resume_text}
            Question:
            {question}
            Answer
        """
         try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert resume analyzer. Answer the following question based only on the provided resume content."},
                    {"role": "user", "content": f"Resume: {resume_text}"},
                    {"role": "user", "content": f"Question: {question}"}
                ],max_tokens = 200,
                temperature = 0.3
            )
            return response['choices'][0]['message']['content']
         except Exception as e:
            st.error(f"Error with OpenAI API: {e}")
            return "Sorry, I couldn't process your query at the moment."
    
    def process_uploaded_file(self, uploaded_file):
        if uploaded_file.type == "application/pdf":
            doc_content = self.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc_content = self.extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a PDF or DOCX.")
            return None, None
        if doc_content:
            preprocess_doc_content = self.preprocess_text(doc_content)
            return doc_content, preprocess_doc_content
        else:
             return None, None
    
    def run(self):
        with st.container():
            input_jd = st.text_input("Job Description:", key="input", placeholder="Enter the job description here...")
            if input_jd:
                st.session_state.preprocess_JD = self.preprocess_text(input_jd)
            else:
                st.error('Please Mention Job Description! ')
            uploaded_file = st.file_uploader("Upload your resume:", type=["pdf", "docx"])
            if uploaded_file:
                pass
            else:
                st.error('Please Upload your Resume..')
            col1, col2, col3 = st.columns(3)
            with col1:
                submit1 = st.button("Submit")
            with col2:
                submit2 = st.button("Generate Cover Letter")
            with col3:
                submit3 = st.button('Chat with Your Resume')
            if submit1:
                if uploaded_file:
                    self.save_resume_file(uploaded_file)
                    doc_content, preprocess_doc_content = self.process_uploaded_file(uploaded_file)
                    if preprocess_doc_content:
                        if input_jd:
                            self.display_skills_and_similarity(st.session_state.preprocess_JD, preprocess_doc_content)
                            response = self.get_openai_response(st.session_state.preprocess_JD, doc_content, self.input_prompt1)
                            st.write(response)
                        else:
                            st.error('Please Mention Job Description')

            if submit2:
                 if uploaded_file:
                     self.save_resume_file(uploaded_file)
                     doc_content, preprocess_doc_content = self.process_uploaded_file(uploaded_file)
                     if preprocess_doc_content:
                         if input_jd:
                            ai_agents_cover_letter = self.generate_cover_letter(input_jd,doc_content)
                            st.subheader("Generated Cover Letter :")
                            st.write(ai_agents_cover_letter)
                         else:
                             st.error("Please mention Job Description..")
                 else:
                     st.error('Please Upload your resume')
            
            if submit3:
                if uploaded_file:
                    doc_content, preprocess_doc_content = self.process_uploaded_file(uploaded_file)
                    st.session_state.doc_content = doc_content
                if st.session_state.get('doc_content', None):
                   user_query = st.text_input("Ask a question about your resume")
                   send_button = st.button('Send Query..',
                   key='send_query_btn',
                   help='Click to send your query',
                   type='primary',
                    )
                   if user_query and send_button:
                        response = self.ask_openai(user_query, st.session_state.doc_content)
                        st.markdown("""
                <style>
                .response-container {
                    background-color: #1e3d59;
                    border-radius: 10px;
                    padding: 5px;
                    margin: 10px 0;
                    color: #ffffff;
                    border-left: 5px solid #17b978;
                }
                </style>
            """, unsafe_allow_html=True)
            try:
                if submit1:
                    text_output = preprocess_doc_content
                    NAME_REGEX = r"^[A-Za-z]+(?: [A-Za-z]+)+$|\b[a-zA-Z]+\s+[a-zA-Z]+\b"
                    PHONE_REGEX = r"(\+91[\s\-]?)?(\d{5})[\s\-]?(\d{2,8})|(\+?\d{1,3})?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10}"
                    EMAIL_REGEX = r"\b[a-z0-9_.]{2,30}[@][gmail.com]{1,15}\b|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                    EXPERIENCE_REGEX = r'(\d+\.?\d*)\s+(years|yrs|year|yr)|(\d+(\.\d+)?)\s*(years|yrs|year|yr)|(\d+)\s*[-\s]?\s*(year|years|yrs|yr)'
                    name_match = re.match(NAME_REGEX, ' '.join(text_output.split()[:2]))
                    if name_match:
                       name = name_match.group()
                    else:
                        name = 'None'
                    phone_match = re.search(PHONE_REGEX, text_output)
                    if phone_match:
                        phone = phone_match.group()
                    else:
                        phone = 'None'
                    email_match = re.search(EMAIL_REGEX, text_output)
                    if email_match:
                        email = email_match.group()
                    else:
                        email = 'None'
                    experience_match = re.search(EXPERIENCE_REGEX, text_output)
                    if experience_match:
                        experience = experience_match.group()
                    else:
                        experience='Fresher'
                    new_df_data = {
                        'Name': [name],
                        'Phone No': [phone],
                        'Email': [email],
                        'Experience': [experience],
                        "JD Skills": [', '.join(self.jd_skills_normalized) if self.jd_skills_normalized else 'None'],
                        "Resume Skills": [', '.join(self.resume_skills_normalized) if self.resume_skills_normalized else 'None'],
                        "Matching Skills between JD & Resume": [', '.join(self.matching_skills) if self.matching_skills else 'None'],
                        "Cosine Similarity": [f"{self.similarity_score * 100:.2f}%" if 'similarity_score' in globals() else 0]
                    }
                    df_detail = pd.DataFrame(new_df_data)
                    file_path = r"D:\LLM_ALL_COLLAB_FOLDERS_freecodecamp_\prathmesh_GenAI_PROJECTS\Resume_Parsing NLP+Gen AI PROJECT\recruitment_data.csv"  #set csv path here
                    df_detail.to_csv(file_path, mode='a', header=not pd.io.common.file_exists(file_path), index=False)
                    st.success(f"Data has been saved to Recruitment_CSV file successfully.")
            except Exception as e:
               st.error(f"Error saving data to CSV: {e}")

if __name__ == '__main__':
    ats_system = ATS_System()
    ats_system.run()
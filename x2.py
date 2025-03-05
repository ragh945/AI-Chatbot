import streamlit as st
import openai
import PyPDF2
import speech_recognition as sr
import random
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import whisper
import sounddevice as sd
import numpy as np
import wave

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI API Key (Use Streamlit secrets for security)
openai_api_key = st.secrets["openai"]["api_key"]

# Use the API key
openai.api_key = openai_api_key

# Predefined PDFs for Python and Machine Learning
pdf_files = {
    "Python": "Python_Concepts.pdf",
    "Machine Learning": "Machine Learning Interview Questions with Answers.pdf"
}

def extract_text_from_pdf(pdf_path, max_chars=5000):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text[:max_chars]

def generate_questions(text, num_questions=5):
    prompt = f"""
    Extract {num_questions} unique interview questions from the following text:
    {text}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        questions = response.choices[0].message.content.split("\n")
        return random.sample(questions, min(num_questions, len(questions)))
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []


def evaluate_answer(user_answer, correct_answer):
    embedding1 = model.encode(user_answer, convert_to_tensor=True)
    embedding2 = model.encode(correct_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    
    keywords = set(correct_answer.lower().split())
    user_keywords = set(user_answer.lower().split())
    keyword_match = len(user_keywords & keywords) / len(keywords) if keywords else 0
    
    score = round((similarity * 0.7 + keyword_match * 0.3) * 5, 2)
    
    return score


def record_audio(filename="audio.wav", duration=5, samplerate=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording complete.")

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

record_audio()



def transcribe_audio(audio_file="audio.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]


def generate_report(responses, total_score, final_result, user_name):
    df = pd.DataFrame(responses)
    df["Max Score"] = 5  # Max score per question

    st.write("### Final Interview Report")
    st.write(f"**Candidate Name:** {user_name}")
    st.table(df)

    st.write(f"**Total Score:** {total_score}/25")
    st.write(f"**Final Result:** {final_result}")

    output = BytesIO()
    c = canvas.Canvas(output, pagesize=letter)

    c.drawString(100, 750, "AI Interview Report")
    c.drawString(100, 730, f"Candidate Name: {user_name}")
    c.drawString(100, 710, f"Total Score: {total_score}/25")
    c.drawString(100, 690, f"Final Result: {final_result}")

    y_position = 670
    for _, row in df.iterrows():
        c.drawString(100, y_position, f"Q: {row['question']}")
        c.drawString(100, y_position - 20, f"Your Answer: {row['answer']}")
        c.drawString(100, y_position - 40, f"Score: {row['score']}/5")
        y_position -= 60

    c.save()
    output.seek(0)
    return output

# Streamlit UI
st.title("🤖 AI Interview Chatbot - Friday")
st.write("Hi! My name is Friday, your AI interviewer. Let's get started!")

# Get User Name
if "user_name" not in st.session_state:
    user_input = st.text_input("Please enter your name:")
    if user_input:
        st.session_state["user_name"] = user_input
        st.success(f"Hi {user_input}, nice to meet you!")
        st.rerun()
else:
    user_name = st.session_state["user_name"]
    st.write(f"Welcome back, {user_name}!")

# Course Selection
topic = st.selectbox("Select Course", ["Python", "Machine Learning"])

if "questions" not in st.session_state:
    st.session_state.update({
        "questions": [],
        "current_question": 0,
        "score": 0,
        "responses": [],
        "incorrect_count": 0
    })

if topic in pdf_files and not st.session_state["questions"]:
    pdf_path = pdf_files[topic]
    text = extract_text_from_pdf(pdf_path)
    st.session_state["questions"] = generate_questions(text)
    st.success("Questions Generated!")

if st.session_state["current_question"] < 5 and st.session_state["incorrect_count"] < 3:
    q_index = st.session_state["current_question"]
    question = st.session_state["questions"][q_index]
    st.write(f"Q{q_index+1}: {question}")

    input_method = st.radio("Answer Input Method", ("Text", "Voice"))
    user_answer = ""

    if input_method == "Text":
        user_answer = st.text_area("Your Answer")
    else:
        if st.button("Record Answer"):
            user_answer = transcribe_audio()
            st.write(f"You said: {user_answer}")

    if st.button("Submit Answer") and user_answer:
        score = evaluate_answer(user_answer, question)

        st.session_state["responses"].append({"question": question, "answer": user_answer, "score": score})
        st.session_state["score"] += score

        if score < 2.5:
            st.session_state["incorrect_count"] += 1

        st.session_state["current_question"] += 1
        st.rerun()

if st.session_state["current_question"] == 5 or st.session_state["incorrect_count"] == 3:
    total_score = round(sum([resp["score"] for resp in st.session_state["responses"]]), 2)
    final_result = "Pass" if total_score >= 13 else "Fail"

    report = generate_report(st.session_state["responses"], total_score, final_result, user_name)
    st.download_button("Download Report", report, "interview_report.pdf", "application/pdf")
    st.stop()

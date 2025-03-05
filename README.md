# 🤖 AI Interview Chatbot - "Friday"
An AI-powered chatbot that conducts interviews using voice and text-based inputs, evaluates responses, and generates performance reports.

## Objective:
The goal of this project is to create an AI-driven interview chatbot, "Friday," that:
✅ Extracts interview questions from a PDF related to Python or Machine Learning.
✅ Asks the candidate random questions from the extracted content.
✅ Accepts voice and text-based answers from the user.
✅ Evaluates answers using semantic similarity and keyword matching.
✅ Maintains a scoreboard and terminates if 3 incorrect answers are given.
✅ Generates a detailed interview report with feedback and downloadable results.
✅ Deploys the chatbot on Streamlit for an interactive experience.

## Libraries Used:
This project integrates multiple AI and NLP-based libraries:
- Streamlit → Web-based UI for chatbot deployment.
- PyPDF2 → Extract text from PDF documents.
- Groq (Gemma2-9b-it) → Generates interview questions from extracted text.
- SpeechRecognition → Captures and transcribes spoken answers.
- SentenceTransformers → Computes semantic similarity between user answers and expected responses.
- Pandas → Handles data storage for user responses.
- ReportLab → Creates PDF reports summarizing user performance.

## Workflow of the Project:
Below is the step-by-step working process of the chatbot:
-  User Interaction
🔹 The user selects a topic (Python or Machine Learning).
🔹 The chatbot extracts questions from a preloaded PDF.
🔹 Five questions are randomly picked and asked to the user.

  - Answer Collection
🔹 The user chooses to answer via Text or Voice.
🔹 If text input, the user types the answer in a text box.
🔹 If voice input, the chatbot records and transcribes the user's response.

  -  Answer Evaluation
🔹 The chatbot evaluates answers based on two factors:

  - Semantic Similarity → Measures how similar the answer is to an ideal response.
Keyword Matching → Checks how many relevant keywords are used.
🔹 A score (0-5) is assigned based on evaluation.

-  Scoring and Termination
🔹 If the user makes 3 incorrect attempts, the interview ends early.
🔹 Otherwise, the chatbot asks all 5 questions and calculates the total score.

-  Report Generation
🔹 A final report is generated with:

  - Candidate Name
Score Breakdown for Each Question
Total Score and Final Verdict (Pass/Fail)
🔹 The PDF report is downloadable for future reference.

## Challenges Faced:
Here are some challenges and the solutions applied in the project:
- 1️⃣ Speech Recognition Accuracy
🔸 Problem: Background noise and accents affect transcription accuracy.
🔸 Solution: Implemented Google Speech API (can be replaced with Whisper AI for better accuracy).

- 2️⃣ Answer Evaluation Complexity
🔸 Problem: Some answers may be partially correct but phrased differently.
🔸 Solution: Used Sentence Transformers (SBERT) for semantic similarity matching.

- 3️⃣ Question Extraction from PDFs
🔸 Problem: Some PDFs do not contain clean text data.
🔸 Solution: Used PyPDF2 to extract content and applied Groq LLM to generate structured questions.

- 4️⃣ Performance Optimization
🔸 Problem: Running multiple API calls (speech-to-text + evaluation) introduces latency.
🔸 Solution: Cached questions, batch-processed answers, and optimized API calls.

## Findings & Summary:
- 🔹 Speech-to-text conversion is effective but can be improved with Whisper AI.
- 🔹 Semantic similarity evaluation helps in grading open-ended responses accurately.
- 🔹 The chatbot successfully mimics a real interview by allowing both voice and text responses.
- 🔹 Early termination logic (3 incorrect answers) ensures efficient interview flow.
- 🔹 Automated report generation adds value for users to track progress.

 ## Screenshots:
 ![image](https://github.com/user-attachments/assets/fc02f057-1847-484f-b3ca-aa37bdaf05f2)

 ## Deployment Link : https://ai-chatbot-w2if96jd6ca9bkpmxqkuh8.streamlit.app/


## Conclusion:
This AI-powered interview chatbot, "Friday," effectively simulates real-world technical interviews by:
- ✅ Asking AI-generated questions from course PDFs.
- ✅ Evaluating answers intelligently using NLP & LLMs.
- ✅ Providing a performance report with scores and feedback.
- ✅ Enhancing user experience with voice-enabled interactions.




import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
from dotenv import load_dotenv

load_dotenv()

from typing import List

def get_pdf_text(pdf_docs):
    text = ""
    # Use default PDF if no file is uploaded
    if not pdf_docs:
        pdf_docs = ["files/science.pdf"]
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    You need to prepare a question paper based on the classes and subjects as well as difficulty level as well as question type
    here are few terms that need to be taken tare while generating questions.
1. Level of Difficulty: Choose from easy, medium, or difficult.
2. Number of Questions: Specify how many questions you'd like in your test paper.
3. Question Type: Select the type of questions you want to include. Here are examples of each type:
    - Long Answer: Questions that require detailed explanations or essays.
        Example: "Explain the process of photosynthesis and its significance in plants."
    - Short Answer: Questions with concise responses.
        Example: "Define osmosis."
    - Multiple Choice: Questions with multiple options, where only one is correct.
        Example: "Which of the following is not a primary color of light? (A) Red (B) Green (C) Yellow (D) Blue"
    - Single Correct: Questions with a single correct answer.
        Example: "What is the chemical symbol for iron?"
    - Fill in the Blanks: Questions where students must complete the missing parts.
        Example: "During cellular respiration, glucose is broken down into ____________ and ____________."

4. Chapter: Specify the chapter from which the questions should be derived.

5. Topics (Optional): If you have specific topics within the chapter, you can mention them here to focus the questions accordingly.

Ensure accuracy in your inputs. If anything is unclear, please provide additional details or clarify your requirements.

---

**Sample Input Case:**
 Chapter: Components of Food
 Difficulty Level: Medium
 Question Type: Long Answer
 Number of Questions: 5

---

**Sample Output Case:**

1. Explain the process of digestion in the human body, highlighting the role of enzymes in breaking down food components.
2. Discuss the importance of balanced diet and its impact on overall health and well-being.
3. Describe the structure and function of carbohydrates, proteins, and fats in the diet, and their role in providing energy to the body.
4. Analyze the concept of food adulteration and its consequences on human health, suggesting preventive measures.
5. Explore the significance of vitamins and minerals in maintaining various physiological functions in the body, citing examples.

---

**Sample Input Case:**
 Chapter: Separation of Substances
 Difficulty Level: Easy
 Question Type: Short Answer
 Number of Questions: 5

---

**Sample Output Case:**

1. Define filtration and provide an example of its application in everyday life.
2. Explain the process of evaporation and describe one practical use of this phenomenon.
3. Describe the method of sieving and identify a situation where sieving is commonly used.
4. Discuss the concept of sedimentation and clarify how it is utilized in the purification of drinking water.
5. Define sublimation and provide an example of a substance that undergoes sublimation under normal conditions.
---

**Sample Input Case:**
Chapter: Motion and Measurement of Distances
Difficulty Level: Medium
Question Type: Multiple Choice
Number of Questions: 5

---

**Sample Output Case:**

You have requested to generate a test paper with the following specifications:
- Chapter: Motion and Measurement of Distances
- Difficulty Level: Medium
- Question Type: Multiple Choice
- Number of Questions: 5

Based on your inputs, here are the questions generated for your test paper:

1. What is the SI unit of length?
   a) Meter
   b) Kilogram
   c) Second
   d) Kelvin

2. What is the formula to calculate speed?
   a) Speed = Distance × Time
   b) Speed = Time / Distance
   c) Speed = Distance / Time
   d) Speed = Time - Distance

3. Which of the following is a scalar quantity?
   a) Velocity
   b) Acceleration
   c) Distance
   d) Force

4. If a car travels 200 meters in 20 seconds, what is its speed?
   a) 10 m/s
   b) 20 m/s
   c) 30 m/s
   d) 40 m/s

5. Which of the following is a vector quantity?
   a) Temperature
   b) Mass
   c) Speed
   d) Displacement

---
Ensure accuracy in your inputs. If anything is unclear, please provide additional details or clarify your requirements.\n\n

Class: {selected_class}
Subject: {selected_subject}
Chapter: {selected_chapter}
Difficulty Level: {selected_difficulty}
Question Type: {selected_question_type}
Number of Questions: {selected_num_questions}

Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=[ "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain



def user_input(selected_class, selected_subject, selected_chapter, selected_difficulty, selected_question_type, selected_num_questions):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(selected_chapter)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "selected_class": selected_class, "selected_subject": selected_subject,"selected_chapter": selected_chapter, "selected_difficulty": selected_difficulty, "selected_question_type": selected_question_type, "selected_num_questions": selected_num_questions}, return_only_outputs=True)
    return response["output_text"]

def get_chapters_from_class(selected_class):
  """
  This function retrieves chapters for a given class from data.json
  """
  try:
    with open("data.json", "r") as file:
      data = json.load(file)
      class_data = data["classes"].get(str(selected_class))
      if class_data:
        return list(class_data["subjects"].keys())  # Return subject keys as chapters
      else:
        return []
  except FileNotFoundError:
    print("Error: data.json not found!")
    return []
  except json.JSONDecodeError:
    print("Error: Invalid JSON format in data.json!")
    return []

def get_subjects_from_class(selected_class):
  """
  This function retrieves subjects for a given class from data.json
  """
  try:
    with open("data.json", "r") as file:
      data = json.load(file)
      class_data = data["classes"].get(str(selected_class))
      if class_data:
        return list(class_data["subjects"].keys())  # Return subject keys
      else:
        return []
  except FileNotFoundError:
    print("Error: data.json not found!")
    return []
  except json.JSONDecodeError:
    print("Error: Invalid JSON format in data.json!")
    return []
def load_data(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data
def get_chapters_from_subject(selected_class, selected_subject):
  """
  This function retrieves chapters for a given subject within a class from data.json
  """
  try:
    with open("data.json", "r") as file:
      data = json.load(file)
      class_data = data["classes"].get(str(selected_class))
      if class_data and selected_subject in class_data["subjects"]:
        return class_data["subjects"][selected_subject]
      else:
        return []
  except FileNotFoundError:
    print("Error: data.json not found!")
    return []
  except json.JSONDecodeError:
    print("Error: Invalid JSON format in data.json!")
    return []
  

def main():
    st.set_page_config("TestGenie")
    st.header("TestGenie")
    data = load_data("data.json")
    # # Add dropdown for selecting class
    selected_class = st.selectbox("Select Class", list(data["classes"].keys()))

    subjects = get_subjects_from_class(selected_class)
    if not subjects:
        st.error("No subjects found for selected class!")
        return

    # Add dropdown for selecting subject
    selected_subject = st.selectbox("Select Subject", subjects)

    # Get chapters based on selected class and subject
    chapters = get_chapters_from_subject(selected_class, selected_subject)
    if not chapters:
        st.error("No chapters found for selected class and subject combination!")
        return

    # Add dropdown for selecting chapter (use chapters list retrieved)
    selected_chapter = st.selectbox("Select Chapter", chapters)

    # Add dropdown for selecting difficulty
    selected_difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

    # Add dropdown for selecting question type
    selected_question_type = st.selectbox("Question Type", [
        "Long Answer Type",
        "Short Answer Type",
        "Multiple Correct Question",
        "Single Correct Question",
        "Fill in the blanks"
    ])

    # Add dropdown for selecting number of questions
    selected_num_questions = st.selectbox("Number of Questions", ["5", "10", "15", "20"])

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            # raw_text = get_pdf_text([])
            # text_chunks = get_text_chunks(raw_text)
            # get_vector_store(text_chunks)
            # st.success("Done")
            result = user_input(selected_class, selected_subject, selected_chapter, selected_difficulty, selected_question_type, selected_num_questions)
            st.write("Reply: ", result)

if __name__ == "__main__":
    main()

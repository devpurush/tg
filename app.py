import streamlit as st
import google.generativeai as genai
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config("TestGenie")
st.header("TestGenie")

try:
    genai.configure(api_key = "AIzaSyAcTc43mWEvHeb_58ln_e-m2tm4nDHSUqw")
except AttributeError as e:
    st.warning("Something went wrong with API Key")

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat()

def user_input(selected_class, selected_subject, selected_chapter, selected_difficulty, selected_question_type, selected_num_questions):
    prompt = f"""
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
    response = chat.send_message(prompt)
    return response.text

def load_data(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def main():
    data = load_data("data.json")

    selected_class = st.selectbox("Select Class", list(data["classes"].keys()))

    subjects = list(data["classes"][selected_class]["subjects"].keys())
    selected_subject = st.selectbox("Select Subject", subjects)

    chapters = data["classes"][selected_class]["subjects"][selected_subject]
    selected_chapter = st.selectbox("Select Chapter", chapters)

    selected_difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

    selected_question_type = st.selectbox("Question Type", [
        "Long Answer Type",
        "Short Answer Type",
        "Multiple Correct Question",
        "Single Correct Question",
        "Fill in the blanks"
    ])

    selected_num_questions = st.selectbox("Number of Questions", ["5", "10", "15", "20"])

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            result = user_input(selected_class, selected_subject, selected_chapter, selected_difficulty, selected_question_type, selected_num_questions)
            st.write("Reply: ", result)

if __name__ == "__main__":
    main()

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def generate_answer(question, context):

    prompt = f"""
You are a helpful assistant answering questions based on the provided document.

Use ONLY the information in the context below to answer the question.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":

    question = "What is a variable in Python?"

    context = """
    A variable is a name used to store a value in Python.
    Python variables do not need to be explicitly declared.
    """

    answer = generate_answer(question, context)

    print("\nAnswer:")
    print(answer)
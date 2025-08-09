from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai

genai.configure(api_key="AIzaSyBF5ys8AH4ZWeyvy1oVHj8NRIVcDA9ad_I")
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")

    response = model.generate_content(question)
    answer = response.text

    return {"answer": answer}
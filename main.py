import os # access env variables
import json
from dotenv import load_dotenv #load variable from .env file
from fastapi import FastAPI,HTTPException # error handling
from google import genai # Gemini SDK lets to talk to Gemini api
from pydantic import BaseModel,field_validator,ValidationError

load_dotenv() #read .env file

app = FastAPI()
# with this we connect to gemini
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
    )

class Request(BaseModel):
    question : str

    @field_validator("question") #custom validator
    @classmethod
    def validate_question(cls,value:str):
        if not value.strip(): #remove extra spaces from front and back
            raise ValueError("Question cannot be empty")
        return value


class Response(BaseModel):
    answer : str

class UserProfile(BaseModel):
    name : str
    experience_years : int
    skills : list[str]

@app.post("/extract-profile", response_model=UserProfile)
def extract_profile(text:str):
    prompt = f"""
Extract the following information from the text.
Return ONLY valid JSON.
Do not incluce Markdown code fences.

Required Fields:
    name (string)
    experience_years (Integer)
    skills (array of strings)
Text: 
{text}
"""
    
    response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt)
    try: 
        data = json.loads(response.text) # convert json string to python object
        return UserProfile(**data)
    except json.JSONDecodeError: #ai doesnot return json string
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid JSON"
        )
    except ValidationError: #raise when failed to pydantic model mismatch data types or miising keys
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid data format"
        )


@app.post("/ask", response_model=Response)
def ask_ai(req:Request):
    system_prompt = """
You are an Ai engineering mentor for beginner developers.
You specialize in python backend development with FASTAPI and Ai-engineering fundamentals.


Explain concepts simply.
Use practical examples.
keep response concise.

"""
    prompt = f"""
{system_prompt}
User question : {req.question}




"""
    try:
        # if connection failed or authentication failed python go to except block
        res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
        )
        # if response is empty
        if not res.text:
            raise HTTPException(
                status_code=502,
                detail="AI service returned an empty response"
            )
        return Response(answer=res.text)
        

    except Exception:
        raise HTTPException(
            status_code=503,
            detail="AI service unavailable"
        )
    
    
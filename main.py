from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from detoxify import Detoxify
import numpy as np
import os

# Initialize FastAPI
app = FastAPI()

# Define the request body model
class Comment(BaseModel):
    text: str

# Load the Detoxify model
model = Detoxify('unbiased')

@app.post("/analyze/")
async def analyze_comment(comment: Comment):
    # Analyze the comment using Detoxify
    results = model.predict(comment.text)
    
    # Convert any numpy.float32 values to Python floats for JSON serialization
    results = {key: float(value) if isinstance(value, np.float32) else value for key, value in results.items()}
    
    return results

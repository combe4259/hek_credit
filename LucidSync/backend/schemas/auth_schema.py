from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class RegisterIn(BaseModel):
    user_id: str = Field(min_length=3, max_length=50)
    phone_number: str = Field(min_length=5, max_length=20)
    password: str = Field(min_length=6)

class LoginIn(BaseModel):
    phone_number: str = Field(min_length=1)
    password: str = Field(min_length=1)

class UserOut(BaseModel):
    user_id: str
    phone_number: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
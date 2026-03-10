from datetime import datetime
from pydantic import BaseModel, EmailStr



class UserCreate(BaseModel):
    username: str
    email: EmailStr
    role_id: int
    created_by: str | None

class UserUpdate(BaseModel):
    id: int
    username: str
    email: EmailStr
    role_id: int
    created_by: str | None


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    role_id: int
    created_by: str | None
    created_on: datetime

    model_config = {"from_attributes": True}

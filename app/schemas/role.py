from datetime import datetime

from pydantic import BaseModel


class RoleCreate(BaseModel):
    rolename: str
    created_by: str | None = None

class RoleUpdate(BaseModel):
    rolename: str | None = None
    created_by: str | None = None    


class RoleResponse(BaseModel):
    id: int
    rolename: str
    created_by: str | None
    created_on: datetime

    model_config = {"from_attributes": True}

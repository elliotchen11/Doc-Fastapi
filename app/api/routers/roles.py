from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.schemas.role import RoleCreate, RoleUpdate, RoleResponse
from app.services.role_service import create_role, list_roles, get_role_by_id, update_role, delete_role

router = APIRouter()


@router.get("/", response_model=list[RoleResponse])
def get_roles(db: Session = Depends(get_db)):
    return list_roles(db)


@router.post("/", response_model=RoleResponse, status_code=201)
def post_role(data: RoleCreate, db: Session = Depends(get_db)):
    return create_role(db, data)

@router.get("/{id}", response_model=RoleResponse)
def get_role(id: int, db: Session = Depends(get_db)):
    role = get_role_by_id(db, id)
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    return role

@router.patch("/{id}", status_code=204)
def patch_role(id: int, data: RoleUpdate, db: Session = Depends(get_db)):
    role = get_role_by_id(db, id)
    if role is None:
        raise HTTPException(status_code=404, detail="Role not found")
    return update_role(db, role, data)

@router.delete("/{id}", status_code=204)
def delete_role(id: int, db: Session = Depends(get_db)):    
    if not delete_role(db, id):
        raise HTTPException(status_code=404, detail="Role not found")
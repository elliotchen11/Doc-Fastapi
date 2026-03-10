from sqlalchemy.orm import Session

from app.crud.role import create_role as crud_create_role
from app.crud.role import get_all_roles
from app.crud.role import get_role_by_id as crud_get_role_by_id
from app.crud.role import update_role as crud_update_role
from app.crud.role import delete_role as crud_delete_role
from app.models.role import Role
from app.schemas.role import RoleCreate
from app.schemas.role import RoleUpdate


def list_roles(db: Session) -> list[Role]:
    return get_all_roles(db)


def create_role(db: Session, data: RoleCreate) -> Role:
    return crud_create_role(db, data)

def get_role_by_id(db: Session, role_id: int) -> Role | None:
    return crud_get_role_by_id(db, role_id)

def update_role(db: Session, role: Role, data: RoleUpdate) -> Role:
    return crud_update_role(db, role, data)

def delete_role(db: Session, role_id: int) -> Role:
    role = crud_delete_role(db, role_id)
    if role is None:
        return False
    return True
from sqlalchemy.orm import Session

from app.models.role import Role
from app.schemas.role import RoleCreate
from app.schemas.role import RoleUpdate


def get_all_roles(db: Session) -> list[Role]:
    return db.query(Role).all()


def create_role(db: Session, data: RoleCreate) -> Role:
    role = Role(**data.model_dump())
    db.add(role)
    db.commit()
    db.refresh(role)
    return role

def get_role_by_id(db: Session, role_id: int) -> Role | None:
    return db.query(Role).filter(Role.id == role_id).first()

def update_role(db: Session, role: Role, data: RoleUpdate) -> Role:
    for field, value in data.model_dump().items():
        setattr(role, field, value)
    db.commit()
    db.refresh(role)
    return role 

def delete_role(db: Session, role_id: int) -> None:
    db.delete(id)
    db.commit()

from sqlalchemy.orm import Session
from sqlalchemy import select
from passlib.hash import bcrypt
from models.user import User

def get_user_by_user_id(db: Session, user_id: str) -> User | None:
    return db.execute(select(User).where(User.user_id == user_id)).scalar_one_or_none()

def create_user(db: Session, user_id: str, phone_number: str, password: str) -> User:
    if get_user_by_user_id(db, user_id):
        raise ValueError("user_id already exists")
    pw_hash = bcrypt.hash(password)
    u = User(user_id=user_id, phone_number=phone_number, password_hash=pw_hash)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def authenticate_user(db, phone_number: str, password: str):
    u = db.execute(
        select(User).where(User.phone_number == phone_number)
    ).scalar_one_or_none()
    if not u:
        return None
    if not bcrypt.verify(password, u.password_hash):
        return None
    return u
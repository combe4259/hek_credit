from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session
from core.settings import settings
from dependies.db_mysql import get_db
from services.user_service import get_user_by_user_id

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def create_access_token(sub: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_exp_minutes)
    payload = {"sub": sub, "exp": exp}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

def get_current_user_id(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(401, "Invalid token")
        return sub
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")

def get_current_user(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    u = get_user_by_user_id(db, user_id)
    if not u:
        raise HTTPException(401, "User not found")
    return u

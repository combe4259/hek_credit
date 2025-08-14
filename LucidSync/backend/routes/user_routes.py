from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from dependies.db_mysql import get_db
from schemas.auth_schema import RegisterIn, LoginIn, UserOut, TokenOut
from services.user_service import create_user, authenticate_user
from dependies.auth import create_access_token, get_current_user
from models.user import User
router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    try:
        u = create_user(db, body.user_id, body.phone_number, body.password)
        return UserOut(user_id=u.user_id, phone_number=u.phone_number)
    except ValueError as e:
        raise HTTPException(409, str(e))


@router.post("/login")
def login(req: LoginIn, db: Session = Depends(get_db)):
    print(f"로그인 요청 데이터: phone_number={req.phone_number}, password={'*' * len(req.password)}")
    user = authenticate_user(db, req.phone_number, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="전화번호 또는 비밀번호가 올바르지 않습니다.")
    
    # JWT 토큰 생성
    access_token = create_access_token(user.user_id)
    
    return {
        "token": access_token,
        "user": {
            "id": user.id,
            "user_id": user.user_id,         
            "phone_number": user.phone_number
        }
    }

@router.post("/logout")
def logout():
    return {"ok": True}

@router.get("/me", response_model=UserOut)
def me(current = Depends(get_current_user)):
    u = current
    return UserOut(user_id=u.user_id, phone_number=u.phone_number)

@router.get("/validate")
def validate_token(current = Depends(get_current_user)):
    return {
        "valid": True,
        "user": {
            "id": current.id,
            "user_id": current.user_id,         
            "phone_number": current.phone_number
        }
    }

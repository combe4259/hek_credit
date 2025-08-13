from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, DateTime, func
from backend.dependies.db_mysql import Base

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)   
    phone_number: Mapped[str] = mapped_column(String(20), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

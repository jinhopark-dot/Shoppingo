from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# --------------------------------------------------------------------------
# 1. DB 연결 설정
#    - "postgresql://사용자명:비밀번호@호스트:포트/DB이름" 형식으로 작성합니다.
#    - 예: "postgresql://user:password@localhost:5432/shoppingo_db"
# --------------------------------------------------------------------------
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:1234@localhost/shoppingo_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
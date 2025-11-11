# app/main.py
# 서버 가동 터미널 입력어: uvicorn app.main:app --reload / uvicorn app.main:app -- reload

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException # Depends, HTTPException 추가
from sqlalchemy.orm import Session # Session 추가
from typing import List # List 추가
from .routers import stores, shopping_lists, routes
from . import crud, models, schemas # crud, models, schemas 임포트
from .database import SessionLocal, engine # SessionLocal, engine 임포트

app = FastAPI(
    title="Shoppingo API",
    description="안녕하세요! RL과 TSP를 활용한 맞춤형 쇼핑 최적 경로 추천 서비스 '쇼핑고'의 백엔드 API입니다.",
    version="1.0.0",
)

# 2. 허용할 출처(origins)를 정의
#    (개발 중에는 "*"로 모든 출처를 허용하는 것이 편리합니다.)
origins = ["*"]

# 3. app에 CORS 미들웨어를 추가합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# DB 세션 의존성 주입 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 상품 ID 리스트로 위치 정보를 반환하는 새 API
@app.post("/api/products/locations", response_model=List[schemas.ProductLocation], tags=["상품 위치 조회"])
def get_product_locations(
    request_data: schemas.ProductLocationRequest, db: Session = Depends(get_db)
):
    """
    상품 ID 리스트 [101, 102, 103]를 입력받아
    각 상품의 위치 노드 정보를 반환합니다.
    """
    products = crud.get_products_locations_by_ids(db, product_ids=request_data.product_ids)
    return products

# "/api" 라는 경로 하위에 stores.py의 API들을 포함시킴
# app.include_router(stores.router, prefix="/api") 
# 라우터들을 앱에 포함
app.include_router(stores.router, prefix="/api", tags=["매장 목록 & 상품"])
app.include_router(shopping_lists.router, prefix="/api", tags=["쇼핑 리스트"]) 
app.include_router(routes.router, prefix="/api", tags=["경로"]) # 새 라우터 추가

@app.get("/")
def read_root():
    return {"message": "Shoppingo API에 오신 것을 환영합니다. 서버는 성공적으로 돌아가고 있습니다."}
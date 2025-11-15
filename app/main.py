# app/main.py
# 서버 가동 터미널 입력어: uvicorn app.main:app --reload / uvicorn app.main:app -- reload
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException, Request # Depends, HTTPException 추가
from fastapi.responses import FileResponse # 이미지 전송을 위한 FileResponse 추가
from sqlalchemy.orm import Session # Session 추가
from typing import List # List 추가
from collections import defaultdict
from .routers import stores, shopping_lists, routes
from . import crud, models, schemas # crud, models, schemas 임포트
from .database import SessionLocal, engine # SessionLocal, engine 임포트
import matplotlib.image as mpimg

from app.ai.get_soluton import load_ai_assets, run_ai_inference
from app.plot import plot_ai_solution

# AI 모델 및 데이터를 한번만 로드 하기 위해서 만든 lifespan 함수 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 서버 시작 시 실행될 코드 (Startup) ---
    print("--- [INFO] 서버 시작: AI 모델 및 데이터 로드 시작 ---")
    
    load_path = 'app/ai/outputs/shopping_30/shopping_run_20251017T155424/best_model.pt'
    full_graph_path = 'app/ai/full_shortest_paths.pkl'
    bk_img_path = 'app/images/map.png'
    
    try:
        background_image = mpimg.imread(bk_img_path)
        app.state.background_image = background_image # app.state에 저장
        print(f"--- [INFO] 배경 이미지 '{bk_img_path}' 로드 완료 ---")
    except Exception as e:
        print(f"--- [CRITICAL] 배경 이미지 로드 실패: {e} ---")
        app.state.background_image = None
    
    # 1. 무거운 함수를 호출하여 모델 로드
    model, opts, full_data = load_ai_assets(load_path, full_graph_path)
    
    # 2. 로드된 자산을 app.state에 저장
    app.state.ai_model = model
    app.state.ai_opts = opts
    app.state.ai_full_data = full_data
    
    print("--- [INFO] AI 자산 로드 완료. API 서버가 준비되었습니다. ---")

    yield # FastAPI 앱이 실행되는 시점

    # --- 서버 종료 시 실행될 코드 (Shutdown) ---
    print("--- [INFO] 서버 종료 ---")
    # app.state에 저장된 객체 참조 제거
    app.state.ai_model = None
    app.state.ai_opts = None
    app.state.ai_full_data = None
    app.state.background_image = None
    
    # GPU 캐시 정리 (PyTorch를 사용 중이므로)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("--- [INFO] GPU 캐시 정리 완료 ---")
    except ImportError:
        pass # torch가 없어도 오류 방지
        
    print("--- [INFO] 자원 정리 완료. 서버를 종료합니다. ---")

app = FastAPI(
    title="Shoppingo API",
    description="안녕하세요! RL과 TSP를 활용한 맞춤형 쇼핑 최적 경로 추천 서비스 '쇼핑고'의 백엔드 API입니다.",
    version="1.0.0",
    lifespan=lifespan
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
    
# "/api" 라는 경로 하위에 stores.py의 API들을 포함시킴
# app.include_router(stores.router, prefix="/api") 
# 라우터들을 앱에 포함
app.include_router(stores.router, prefix="/api", tags=["매장 목록 & 상품"])
app.include_router(shopping_lists.router, prefix="/api", tags=["쇼핑 리스트"]) 
app.include_router(routes.router, prefix="/api", tags=["경로"]) # 새 라우터 추가

@app.get("/")
def read_root():
    return {"message": "Shoppingo API에 오신 것을 환영합니다. 서버는 성공적으로 돌아가고 있습니다."}
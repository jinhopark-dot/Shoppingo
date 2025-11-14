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
        

# IN: ProductListRequest / OUT: RouteResponse --> 둘다 schemas.py에 존재합니다
#   Flutter에서 이 API를 호출할 때는 **"상품 리스트"와 "시작 노드"를 동시에** 넘겨야 합니다 ---> 중요!!
#       따라서 최초 경로 최적화 호출시에는 "시작 노드"를 'S1'으로,
#       그리고 경로 '재'생성 시에는 "시작 노드"를 사용자가 고른 것으로 json 구성해서 넣으면 됩니다
# 성공적으로 구동 테스트 완료 -> 이미지 잘 저장됩니다
@app.post("/api/routes", response_model=schemas.RouteResponse, tags=["최적 경로 생성"]) 
def get_optimal_route(
    request: Request, request_body: schemas.ProductListRequest, db: Session = Depends(get_db)
):
    """
    상품 ID 리스트 (e.g [101, 102, 103])를 입력받아
    경로와 각 상품의 노드를 순서대로 반환합니다.
    """
    # 1. ProductListRequest 스키마에서 product_ids 리스트만 가져오기
    product_list = request_body.product_ids
    start_node = request_body.start_node
    
    start_and_end = [start_node, 'E1'] 
    
    # 2. DB 쿼리
    db_products_data = db.query(
                                models.Product.id, 
                                models.Product.location_node
                        )\
                        .filter(models.Product.id.in_(product_list))\
                        .all()
                        
    # 3. 위치 노드 리스트 변환 (중복 제거)  +  [시작 노드, 끝 노드] 강제 추가
    location_list = list(set([location for (id, location) in db_products_data])) + start_and_end
    
    # AI모델과 pkl 파일 위치
    load_path = 'app/ai/outputs/shopping_30/shopping_run_20251017T155424/best_model.pt'
    full_graph_path = 'app/ai/full_shortest_paths.pkl'
    
    # 4. AI 모델 호출
    ai_result = run_ai_inference(
        model=request.app.state.ai_model,
        opts=request.app.state.ai_opts,
        full_data=request.app.state.ai_full_data,
        shopping_list=location_list,
        start_node_label=start_node,
        num_samples=1000
    )
    if ai_result is None:
        print("[오류] AI 추론이 'None'을 반환했습니다.")
        raise HTTPException(status_code=500, detail="AI 모델 추론에 실패했습니다.")
    
    # 5. [(), (), ()...] 형태로 재가공 (List 안에 List로)
    location_to_products_map = defaultdict(list)
    for prod_id, location in db_products_data:
        location_to_products_map[location].append(prod_id)
        
    final_ordered_list = []
    for location_node in ai_result:
        if location_node == start_node or location_node == 'E1': # 반환할 때는 시작/도착 노드 제외
            continue
        product_ids_for_node = location_to_products_map.get(location_node, [])
        final_ordered_list.append(list(product_ids_for_node))
    

    
    # 6. AI 모델 결과로 Plot, 이미지 생성
    bk_img_path = 'app/images/map.png'
    save_path = 'app/images/ai_route_img.png'
    plot_ai_solution(
            ai_path_labels=ai_result,
            loaded_graph_data=request.app.state.ai_full_data,
            background_image_data=request.app.state.background_image,
            save_filename=save_path
        )
    
    image_url = "/api/routes/image" # 이미지 Url을 묶어서 response

    ai_result = ai_result[1:-1]
    # 총 3가지 반환
    return schemas.RouteResponse(
        ordered_node = ai_result, # ai가 산출한 노드 시퀀스
        ordered_product_ids=final_ordered_list, # 각 노드당 사야 하는 product id, [[], [], []...] 형태
        route_image_url=image_url # 이미지 url
    )

# 새로운 API : 이미지 반환 API
@app.get("/api/routes/image", tags=["최적 경로 이미지 조회"])
async def get_route_image():
    """
    /api/routes 에서 생성된 'ai_route_img.png' 파일을 반환합니다.
    """
    # 1. API 1에서 저장한 이미지 파일의 경로
    image_path = 'app/images/ai_route_img.png'
    
    if not os.path.exists(image_path):
        print(f"[오류] /api/routes/image: '{image_path}'에서 파일을 찾을 수 없습니다.")
        raise HTTPException(status_code=404, detail="Image not found")
    
    # 2. FileResponse를 사용해 해당 파일을 클라이언트(Flutter)에 전송
    #    (파일이 없으면 404 에러는 FastAPI가 자동으로 처리해줌)
    return FileResponse(image_path, media_type="image/png")

#----------------------------------------------------------------------------------------------------------------
# ----- 이미지 반환 과정 ------ (규헌이 형은 보세요)
#    1단계 : get_optimal_route에서 새로 정의된 스키마인 RouteResponse 형식으로, 리스트와 "이미지 URL"을 보냄
#    2단계 : 프론트 쪽에서 get_route_image(엔드포인트 : /api/routes/image)를 호출하면 이미지를 보냄
#       *참고사항 : 이미지는 무조건 한장만 저장되기에 여러명이 동시에 사용하면 문제가 있지만, 
#                   데모용이기 때문에 그냥 하나의 이미지(ai_route_img.png)로만 관리

# ----- Flutter에서 호출하는 방법 ----- (규헌이 형은 보세요)
#    1. Flutter가  /api/routes를 호출
#    2. JSON 응답을 받는다 (RouteResponse 스키마 참고)
#    3. Flutter의 Image.network()가 response.route_image_url ("/api/routes/image")을 호출 
#                -> 예시: Image.network("http://YOUR_SERVER_IP:8000/api/routes/image")
#    4. 이렇게 하면 서버의 /api/routes/image API가 실행되어 app/images/ai_route_img.png 파일을 찾아 Flutter에 전송
#      * 자세한 사항은 잼민이와 함께...
#----------------------------------------------------------------------------------------------------------------
    
# "/api" 라는 경로 하위에 stores.py의 API들을 포함시킴
# app.include_router(stores.router, prefix="/api") 
# 라우터들을 앱에 포함
app.include_router(stores.router, prefix="/api", tags=["매장 목록 & 상품"])
app.include_router(shopping_lists.router, prefix="/api", tags=["쇼핑 리스트"]) 
app.include_router(routes.router, prefix="/api", tags=["경로"]) # 새 라우터 추가

@app.get("/")
def read_root():
    return {"message": "Shoppingo API에 오신 것을 환영합니다. 서버는 성공적으로 돌아가고 있습니다."}
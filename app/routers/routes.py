# app/routers/routes.py
import os
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, Request 
from fastapi.responses import FileResponse 
from sqlalchemy.orm import Session 
import time

from .. import crud, models, schemas # 상위 디렉토리의 crud, models, schemas 임포트
from ..database import SessionLocal # 상위 디렉토리의 SessionLocal 임포트
from ..ai.get_soluton import run_ai_inference # AI 함수 임포트
from ..plot import plot_ai_solution # Plot 함수 임포트

router = APIRouter()

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
@router.post("/routes", response_model=schemas.RouteResponse, tags=["최적 경로 생성"]) 
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
    start_product_id_int = None
    
    # 'S1'이 아니라 101, 111과 같은 상품 명이 들어왔을 때 start node 처리
    if start_node != 'S1':
        try:
            start_product_id_int = int(start_node) # product_id가 들어와야 함
            loacation_data = (db.query(models.Product.location_node).filter(models.Product.id == start_product_id_int).first())
            if loacation_data:
                start_node = loacation_data[0]
        except ValueError:
            pass
    
    is_start_node_in_list = (start_product_id_int is not None) and (start_product_id_int in product_list)
            
        
    start_and_end = [start_node, 'E1']
    
    # 2. DB 쿼리
    db_products_data = db.query(
                                models.Product.id, 
                                models.Product.location_node
                        )\
                        .filter(models.Product.id.in_(product_list))\
                        .all()
                        
    # 3. 위치 노드 리스트 변환 (중복 제거)  +  [시작 노드, 끝 노드] 강제 추가
    product_locations_set = set([location for (id, location) in db_products_data])
    if start_node in product_locations_set:
        product_locations_set.remove(start_node)
    location_list = list(product_locations_set) + start_and_end
    
    # AI모델과 pkl 파일 위치
    load_path = 'app/ai/outputs/shopping_30/shopping_run_20251017T155424/best_model.pt'
    full_graph_path = 'app/ai/full_shortest_paths.pkl'
    
    # 4. AI 모델 호출
    ai_result, best_cost = run_ai_inference(
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
        if location_node == 'E1': # 반환할 때는 시작/도착 노드 제외
            continue
        if location_node == start_node:
            if not is_start_node_in_list:
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
    
    timestamp = int(time.time())
    image_url = f"/api/routes/route_image?t={timestamp}" # 이미지 Url을 묶어서 response

    if is_start_node_in_list:
        ai_result = ai_result[:-1]
    else:
        ai_result = ai_result[1:-1]
    # 총 3가지 반환
    return schemas.RouteResponse(
        ordered_node = ai_result, # ai가 산출한 노드 시퀀스
        ordered_product_ids=final_ordered_list, # 각 노드당 사야 하는 product id, [[], [], []...] 형태
        route_image_url=image_url # 이미지 url
    )

# 새로운 API : 이미지 반환 API
@router.get("/routes/route_image", tags=["최적 경로 이미지 조회"])
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
# ----- 이미지 반환 과정 ------
#    1단계 : get_optimal_route에서 새로 정의된 스키마인 RouteResponse 형식으로, 리스트와 "이미지 URL"을 보냄
#    2단계 : 프론트 쪽에서 get_route_image(엔드포인트 : /api/routes/image)를 호출하면 이미지를 보냄
#       *참고사항 : 이미지는 무조건 한장만 저장되기에 여러명이 동시에 사용하면 문제가 있지만, 
#                   데모용이기 때문에 그냥 하나의 이미지(ai_route_img.png)로만 관리

# ----- Flutter에서 호출하는 방법 ----- 
#    1. Flutter가  /api/routes를 호출
#    2. JSON 응답을 받는다 (RouteResponse 스키마 참고)
#    3. Flutter의 Image.network()가 response.route_image_url ("/api/routes/image")을 호출 
#                -> 예시: Image.network("http://YOUR_SERVER_IP:8000/api/routes/image")
#    4. 이렇게 하면 서버의 /api/routes/image API가 실행되어 app/images/ai_route_img.png 파일을 찾아 Flutter에 전송
#      * 자세한 사항은 잼민이와 함께...
#----------------------------------------------------------------------------------------------------------------
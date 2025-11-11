# app/ai_model_service.py

import httpx  # 1. httpx 임포트
from typing import List
from . import models, schemas
from fastapi import HTTPException # 오류 처리를 위해 임포트

# 2. AI 모델이 있는 컴퓨터의 주소를 입력합니다.
# (예시: AI 컴퓨터의 IP가 192.168.0.50이고 8080 포트에서 실행 중일 경우)
# 이 주소는 AI 팀과 협의하여 확정해야 합니다.
AI_MODEL_SERVER_URL = "http://192.168.0.50:8080/calculate_route"

# 3. AI 모델 호출 함수는 이제 필요 없으므로 삭제합니다.
# def calculate_optimal_route_from_model(node_ids: List[str]) -> List[str]:
#    ... (이 함수 전체 삭제) ...


# 4. get_optimal_route 함수를 'async' 비동기 함수로 변경합니다.
async def get_optimal_route(
    db_list_items: List[models.ListItem], store_layout: dict
) -> dict:
    """
    쇼핑 리스트로 AI 서버에 API 요청을 보내고, 최적 경로를 받아 반환합니다.
    """

    # 1. 중복 제거 로직 (이전과 동일)
    all_product_nodes = [item.product.location_node for item in db_list_items]
    unique_nodes_set = set(all_product_nodes)
    unique_nodes_list = list(unique_nodes_set)

    # 2. httpx.AsyncClient를 사용해 AI 서버에 POST 요청을 보냅니다.
    async with httpx.AsyncClient() as client:
        try:
            # AI 서버에 {"nodes": ["A-01", "B-03"]} 형식의 JSON을 보냅니다.
            response = await client.post(
                AI_MODEL_SERVER_URL, json={"nodes": unique_nodes_list}
            )
            
            # AI 서버가 4xx, 5xx 오류를 반환하면 여기서 에러를 발생시킵니다.
            response.raise_for_status()
            
            # AI 서버가 반환한 JSON 데이터를 파싱합니다.
            # (예: {"optimal_route": ["B-03", "A-01"]})
            data = response.json()
            ordered_nodes = data.get("optimal_route")
            
            if not ordered_nodes:
                raise HTTPException(status_code=500, detail="AI 서버 응답 오류")

        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            # AI 서버에 연결할 수 없거나 오류가 발생한 경우
            print(f"AI 서버 연결 오류: {e}")
            raise HTTPException(
                status_code=503, detail="AI 모델 서버에 연결할 수 없습니다."
            )

    # 3. AI 서버로부터 받은 'ordered_nodes'로 거리/시간 계산 (이전과 동일)
    total_distance = float(len(ordered_nodes) * 15)
    estimated_time = int(len(ordered_nodes) * 2)

    # 4. 최종 결과 반환
    result = {
        "ordered_node_ids": ordered_nodes,
        "total_distance": total_distance,
        "estimated_time": estimated_time,
    }
    return result
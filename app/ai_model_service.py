# app/ai_model_service.py
import sys
import os

# 현재 파일(main.py)의 절대 경로를 가져옵니다.
current_file_path = os.path.abspath(__file__)
# main.py가 있는 디렉토리 (프로젝트 루트) 경로를 가져옵니다.
project_root = os.path.dirname(current_file_path)

# 이 경로를 Python의 모듈 검색 경로 리스트(sys.path)에 추가합니다.
if project_root not in sys.path:
    sys.path.append(project_root)
    
import httpx  # 1. httpx 임포트
from typing import List
from . import models, schemas
from fastapi import HTTPException # 오류 처리를 위해 임포트
from app.ai.get_soluton import get_shopping_route


# 4. get_optimal_route 함수를 'async' 비동기 함수로 변경합니다.
async def get_optimal_route(
    shopping_list, start_node='S1'
) -> list:
    load_path = './ai/outputs/shopping_30/shopping_run_20251017T155424/best_model.pt'
    full_graph_path = './ai/full_shortest_paths.pkl'
    
    result = get_shopping_route(
        load_path=load_path,
        full_graph_path=full_graph_path,
        shopping_list=shopping_list,
        start_node_label=start_node,
        num_samples=1000
    )

    return result
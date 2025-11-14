# app/routers/routes.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload


from .. import crud, models, schemas
from ..database import SessionLocal
from .. import plot

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 1. 함수 정의를 'async def'로 변경합니다.
@router.post("/routes/optimize/{list_id}", response_model=schemas.Route)
async def get_optimized_route(list_id: int, db: Session = Depends(get_db)):
    """
    쇼핑 리스트 ID를 받아 AI 모델을 통해 최적 경로를 생성하는 API
    """
    
    # 2. DB 조회는 동기 방식이므로 'await'가 필요 없습니다. (그대로 둠)
    shopping_list = (
        db.query(models.ShoppingList)
        .options(
            joinedload(models.ShoppingList.items).joinedload(models.ListItem.product)
        )
        .filter(models.ShoppingList.id == list_id)
        .first()
    )

    if not shopping_list:
        raise HTTPException(status_code=404, detail="Shopping list not found")

    if not shopping_list.items:
        raise HTTPException(status_code=400, detail="Shopping list is empty")

    store = crud.get_store(db, store_id=shopping_list.store_id)
    if not store or not store.layout_nodes:
        raise HTTPException(status_code=404, detail="Store layout not found")

    # 3. AI 서비스 호출 부분을 'await'로 변경합니다.
    route_result = await plot.get_optimal_route(
        db_list_items=shopping_list.items, store_layout=store.layout_nodes
    )

    return route_result
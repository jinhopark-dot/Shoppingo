# app/routers/stores.py

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import List

from .. import crud, models, schemas
from ..database import SessionLocal

router = APIRouter()

# Dependency: DB 세션 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --------------------------------------------------------------------------
# [수정됨] response_model을 StoreBase -> StoreInfo로 변경
@router.get("/stores", response_model=List[schemas.StoreInfo])
def read_stores(request: Request, skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
# --------------------------------------------------------------------------
    """모든 매장 목록을 조회하는 API"""
    stores_from_db = crud.get_stores(db, skip=skip, limit=limit)
    stores_with_urls = []
    for store in stores_from_db:
        store_schema = schemas.StoreInfo.model_validate(store)
        store_schema.image_url = f"{str(request.base_url)}static/images/store_img_{store.id}.jpg"
        stores_with_urls.append(store_schema)
    
    return stores_with_urls

@router.get("/stores/{store_id}/products", response_model=List[schemas.Product])
def read_store_products(store_id: int, request: Request, db: Session = Depends(get_db)):
    """특정 매장의 전체 상품 목록을 조회하는 API"""
    db_store = crud.get_store(db, store_id=store_id)
    if db_store is None:
        raise HTTPException(status_code=404, detail="Store not found")
    products_from_db = crud.get_products_by_store(db, store_id=store_id)
    
    products_with_urls = []
    for product in products_from_db:
        product_schema = schemas.Product.model_validate(product)
        product_schema.image_url = f"{str(request.base_url)}static/images/product_img_{product.id}.jpg"
        products_with_urls.append(product_schema)

    return products_with_urls
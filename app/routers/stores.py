# app/routers/stores.py

from fastapi import APIRouter, Depends, HTTPException
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
def read_stores(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
# --------------------------------------------------------------------------
    """모든 매장 목록을 조회하는 API"""
    stores = crud.get_stores(db, skip=skip, limit=limit)
    return stores

@router.get("/stores/{store_id}/products", response_model=List[schemas.Product])
def read_store_products(store_id: int, db: Session = Depends(get_db)):
    """특정 매장의 전체 상품 목록을 조회하는 API"""
    db_store = crud.get_store(db, store_id=store_id)
    if db_store is None:
        raise HTTPException(status_code=404, detail="Store not found")
    products = crud.get_products_by_store(db, store_id=store_id)
    return products
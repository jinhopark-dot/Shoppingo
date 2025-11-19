# app/crud.py

from sqlalchemy.orm import Session 
from typing import List
from . import models
from . import schemas

def get_store(db: Session, store_id: int):
    """ID로 단일 매장 정보 조회"""
    return db.query(models.Store).filter(models.Store.id == store_id).first()

def get_stores(db: Session, skip: int = 0, limit: int = 10):
    """모든 매장 정보 조회"""
    return db.query(models.Store).offset(skip).limit(limit).all()

def get_products_by_store(db: Session, store_id: int):
    """특정 매장의 모든 상품 정보 조회"""
    return db.query(models.Product).filter(models.Product.store_id == store_id).all()

# app/crud.py

def get_products_by_ids(db: Session, product_ids: List[int]):
    """ID 목록을 기반으로 여러 상품의 정보를 조회합니다."""
    return (
        db.query(models.Product)
        .filter(models.Product.id.in_(product_ids))
        .all()
    )
    
def get_products_locations_by_ids(db: Session, product_ids: List[int]):
    """ID 목록을 기반으로 여러 상품의 정보를 조회합니다."""
    
    return list(db.query(models.Product.location_node).filter(models.Product.id.in_(product_ids)).all())

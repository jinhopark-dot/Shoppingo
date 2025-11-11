# app/crud.py

from sqlalchemy.orm import Session, joinedload  # <--- joinedload 임포트 추가
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

# --------------------------------------------------------------------------
# [수정됨] 쇼핑 리스트를 가져올 때, 'items' 항목도 함께 로드(joinedload)
def get_shopping_list(db: Session, list_id: int):
    """ID로 단일 쇼핑 리스트 조회 (항목 포함)"""
    return (
        db.query(models.ShoppingList)
        .options(joinedload(models.ShoppingList.items))  # <--- items를 Eager Loading
        .filter(models.ShoppingList.id == list_id)
        .first()
    )
# --------------------------------------------------------------------------

def create_shopping_list(db: Session, shopping_list: schemas.ShoppingListCreate):
    """쇼핑 리스트 생성 (사용자는 user_id=1로 고정)"""
    db_list = models.ShoppingList(
        user_id=1,
        store_id=shopping_list.store_id,
        name=shopping_list.name,
    )
    db.add(db_list)
    db.commit()
    db.refresh(db_list)
    return db_list

def add_item_to_list(db: Session, item: schemas.ListItemCreate, list_id: int):
    """쇼핑 리스트에 상품 추가"""
    # 1. 먼저, 이 리스트에 같은 상품이 이미 있는지 찾아봅니다.
    existing_item = (
        db.query(models.ListItem)
        .filter(
            models.ListItem.list_id == list_id, 
            models.ListItem.product_id == item.product_id
        )
        .first()
    )

    if existing_item:
        # 2. 이미 존재한다면? -> 기존 수량에 더하기(+)를 합니다.
        existing_item.quantity += item.quantity
        db.commit()
        db.refresh(existing_item)
        return existing_item
    else:
        # 3. 없다면? -> 아까처럼 새로 만듭니다.
        db_item = models.ListItem(**item.model_dump(), list_id=list_id)
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        return db_item
    
# app/crud.py

def get_products_by_ids(db: Session, product_ids: List[int]):
    """ID 목록을 기반으로 여러 상품의 정보를 조회합니다."""
    return (
        db.query(models.Product)
        .filter(models.Product.id.in_(product_ids))
        .all()
    )
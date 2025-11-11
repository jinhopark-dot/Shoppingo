# app/routers/shopping_lists.py

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

@router.post("/shopping-lists", response_model=schemas.ShoppingList)
def create_shopping_list(
    shopping_list: schemas.ShoppingListCreate, db: Session = Depends(get_db)
):
    """새로운 쇼핑 리스트를 생성하는 API (user_id=1 고정)"""
    return crud.create_shopping_list(db=db, shopping_list=shopping_list)


@router.get("/shopping-lists/{list_id}", response_model=schemas.ShoppingList)
def read_shopping_list(list_id: int, db: Session = Depends(get_db)):
    """특정 쇼핑 리스트와 그 안의 모든 상품을 조회하는 API"""
    db_list = crud.get_shopping_list(db, list_id=list_id)
    if db_list is None:
        raise HTTPException(status_code=404, detail="Shopping list not found")
    return db_list


@router.post("/shopping-lists/{list_id}/items", response_model=schemas.ListItem)
def add_item_to_shopping_list(
    list_id: int, item: schemas.ListItemCreate, db: Session = Depends(get_db)
):
    """특정 쇼핑 리스트에 상품을 추가하는 API"""
    db_list = crud.get_shopping_list(db, list_id=list_id)
    if db_list is None:
        raise HTTPException(status_code=404, detail="Shopping list not found")
    return crud.add_item_to_list(db=db, item=item, list_id=list_id)
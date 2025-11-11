# app/schemas.py

from pydantic import BaseModel
from typing import List, Optional

# --- Product Schemas ---
class ProductBase(BaseModel):
    name: str
    location_node: str
    price: int

class Product(ProductBase):
    id: int
    store_id: int

    class Config:
        from_attributes = True

# --- Store Schemas ---
class StoreBase(BaseModel):
    name: str
    address: Optional[str] = None

# --------------------------------------------------------------------------
# [수정됨] GET /api/stores 에서 id도 반환하기 위한 새 스키마
class StoreInfo(StoreBase):
    id: int

    class Config:
        from_attributes = True
# --------------------------------------------------------------------------

class Store(StoreInfo):  # <--- StoreBase가 아닌 StoreInfo를 상속하도록 변경
    products: List[Product] = []

    class Config:
        from_attributes = True

# --- ListItem Schemas ---
class ListItemBase(BaseModel):
    product_id: int
    quantity: int = 1

class ListItemCreate(ListItemBase):
    pass

class ListItem(ListItemBase):
    id: int
    list_id: int

    class Config:
        from_attributes = True

# --- ShoppingList Schemas ---
class ShoppingListBase(BaseModel):
    name: Optional[str] = "My Shopping List"

class ShoppingListCreate(ShoppingListBase):
    store_id: int

class ShoppingList(ShoppingListBase):
    id: int
    user_id: int
    store_id: int
    items: List[ListItem] = []

    class Config:
        from_attributes = True

# --- Route Schemas ---
class Route(BaseModel):
    ordered_node_ids: List[str]
    total_distance: float
    estimated_time: int

    class Config:
        from_attributes = True
        
# --- Product Location Schemas ---

# API에 요청할 때 사용할 스키마
class ProductLocationRequest(BaseModel):
    product_ids: List[int]  # 예: [101, 102, 103]

# API가 응답할 때 사용할 스키마
class ProductLocation(BaseModel):
    id: int
    name: str
    location_node: str

    class Config:
        from_attributes = True
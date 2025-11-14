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
# *수정됨* Location을 조회할 필요가 없으므로 -------> main.get_optimal_route의 입력 스키마
class ProductListRequest(BaseModel):
    product_ids: List[int]  # 예: [101, 102, 103]
    start_node : str # 예: 'S1', 'T{숫자}'

# API가 응답할 때 사용할 스키마 ----------> 이제 필요한가요...?
class ProductLocation(BaseModel):
    id: int
    name: str
    location_node: str

    class Config:
        from_attributes = True
        
        
# ⭐️ /api/routes의 최종 반환 모델 -------> 추가됨!!! main.get_optimal_route에서 반환 스키마
class RouteResponse(BaseModel):
    ordered_node : List[str]
    ordered_product_ids: List[List[int]]  # final_ordered_list
    route_image_url: str                # 생성된 이미지의 URL(엔드포인트)
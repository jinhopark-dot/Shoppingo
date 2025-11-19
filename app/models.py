# from sqlalchemy import (
#     create_engine,
#     Column,
#     Integer,
#     String,
#     Boolean,
#     ForeignKey,
#     TIMESTAMP,
#     Float,
# )
# from sqlalchemy.dialects.postgresql import JSONB
# from sqlalchemy.orm import relationship
# from sqlalchemy.sql import func
# from .database import Base


# # 사용자 계정 관리 테이블 [cite: 157]
# class User(Base):
#     __tablename__ = "users"
    
#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String, unique=True, index=True, nullable=False)
#     hashed_password = Column(String, nullable=False)
#     name = Column(String)
#     created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

#     shopping_lists = relationship("ShoppingList", back_populates="owner")


# # 매장 및 레이아웃 관리 테이블 [cite: 159]
# class Store(Base):
#     __tablename__ = "stores"
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String, nullable=False)
#     address = Column(String)
#     operating_hours = Column(String)
#     # 매장의 모든 위치 노드 좌표 데이터 (행렬 형태) [cite: 160]
#     layout_nodes = Column(JSONB)

#     products = relationship("Product", back_populates="store")


# # 상품 카탈로그 테이블 [cite: 161]
# class Product(Base):
#     __tablename__ = "products"
#     product_id = Column(Integer, primary_key=True, index=True)
#     store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
#     name = Column(String, nullable=False)
#     section = Column(String)
#     # 매장 레이아웃 상의 상품 위치 노드 ID [cite: 162]
#     node_id = Column(String, nullable=False)

#     store = relationship("Store", back_populates="products")


# # 쇼핑 리스트 헤더 테이블 [cite: 163]
# class ShoppingList(Base):
#     __tablename__ = "shopping_lists"
#     list_id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
#     store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
#     name = Column(String, default="My Shopping List")
#     created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

#     owner = relationship("User", back_populates="shopping_lists")
#     items = relationship("ListItem", back_populates="shopping_list")


# # 쇼핑 리스트 항목 테이블 [cite: 165]
# class ListItem(Base):
#     __tablename__ = "list_items"
#     item_id = Column(Integer, primary_key=True, index=True)
#     list_id = Column(Integer, ForeignKey("shopping_lists.list_id"), nullable=False)
#     product_id = Column(Integer, ForeignKey("products.product_id"), nullable=False)
#     quantity = Column(Integer, default=1, nullable=False)
#     is_purchased = Column(Boolean, default=False, nullable=False)

#     shopping_list = relationship("ShoppingList", back_populates="items")

# app/models.py

from sqlalchemy import (
    Column,
    Integer,
    String,
    ForeignKey,
    UniqueConstraint, # Product 테이블의 복합 유니크 제약조건을 위해 필요
    Boolean,          # (만약 다른 테이블에서 쓴다면 유지, 안 쓴다면 삭제 가능)
    TIMESTAMP,        # User, ShoppingList 등에서 사용
    Float             # (사용처가 있다면 유지)
)
from sqlalchemy.dialects.postgresql import JSONB # <--- Store 테이블을 위해 반드시 필요!
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)


class Store(Base):
    __tablename__ = "stores"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    address = Column(String)
    # AI가 사용할 매장의 모든 노드 좌표 데이터 (JSONB 타입)
    layout_nodes = Column(JSONB) 

    products = relationship("Product", back_populates="store")


class Product(Base):
    __tablename__ = "products"
    
    # [변경 1] 실제 DB의 식별자 (PK) - master_id
    master_id = Column(Integer, primary_key=True, index=True)

    # [변경 2] 기존 id는 이제 '상품 코드(SKU)' 역할만 수행 (PK 아님, 중복 허용)
    id = Column(Integer, nullable=False)
    
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    name = Column(String, nullable=False)
    location_node = Column(String, nullable=False)
    price = Column(Integer, nullable=False)

    store = relationship("Store", back_populates="products")
    # list_items 관계 코드는 이미 삭제하셨으므로 없어도 됩니다.

    # [변경 3] DB의 Unique Constraint와 씽크 맞추기
    # "같은 매장 안에서는 같은 상품 코드가 두 개일 수 없다"
    __table_args__ = (
        UniqueConstraint('store_id', 'id', name='unique_store_product'),
    )
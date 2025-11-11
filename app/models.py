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
    Boolean,
    ForeignKey,
    TIMESTAMP,
    Float,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)

    shopping_lists = relationship("ShoppingList", back_populates="owner")


class Store(Base):
    __tablename__ = "stores"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    address = Column(String)
    # AI가 사용할 매장의 모든 노드 좌표 데이터 (JSONB 타입)
    layout_nodes = Column(JSONB) 

    products = relationship("Product", back_populates="store")
    shopping_lists = relationship("ShoppingList", back_populates="store")


class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    name = Column(String, nullable=False)
    location_node = Column(String, nullable=False) # AI가 사용할 위치 노드 ID
    price = Column(Integer, nullable=False)

    store = relationship("Store", back_populates="products")
    list_items = relationship("ListItem", back_populates="product")


class ShoppingList(Base):
    __tablename__ = "shopping_lists"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    store_id = Column(Integer, ForeignKey("stores.id"), nullable=False)
    
    name = Column(String)
    
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    owner = relationship("User", back_populates="shopping_lists")
    store = relationship("Store", back_populates="shopping_lists")
    items = relationship("ListItem", back_populates="shopping_list", cascade="all, delete-orphan")


class ListItem(Base):
    __tablename__ = "list_items"
    
    id = Column(Integer, primary_key=True)
    list_id = Column(Integer, ForeignKey("shopping_lists.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)

    shopping_list = relationship("ShoppingList", back_populates="items")
    product = relationship("Product", back_populates="list_items")
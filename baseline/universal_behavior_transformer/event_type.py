"""
Event type definitions for the Universal Behavioral Modeling Data Challenge.
"""
from enum import Enum

class EventType(Enum):
    """Event types in the dataset."""
    PRODUCT_BUY = 0  # 购买商品
    ADD_TO_CART = 1  # 加入购物车
    REMOVE_FROM_CART = 2  # 从购物车移除
    PAGE_VISIT = 3  # 浏览商品
    SEARCH_QUERY = 4  # 搜索查询 
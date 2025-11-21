import os
import sys
from enum import Enum
from typing import Dict, List, Optional

from fastmcp import FastMCP
from loguru import logger

from sqlalchemy import Column, Float, Integer, String, create_engine, select
from sqlalchemy.orm import Session, declarative_base

logger.remove()
logger.add(sys.stderr, level="INFO")


engine = create_engine("sqlite:///simple_inventory.db", echo=False)
Base = declarative_base()

mcp = FastMCP("product database")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    category = Column(String(50))
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)

    def __repr__(self):
        return f"<Product(id={self.id}, name='{self.name}', price=${self.price:.2f}, stock={self.stock})>"


class ProductCategory(Enum):
    Computers = "Computers"
    Phones = "Phones"
    Audio = "Audio"
    Tablets = "Tablets"
    Wearables = "Wearables"
    All = "All"


# Seed with sample data
def seed_sample_data():
    with Session(engine) as session:
        # Clear existing data
        session.query(Product).delete()

        # Add sample products
        products = [
            Product(name="MacBook Pro", category="Computers", price=1999.99, stock=10),
            Product(name="iPhone 15", category="Phones", price=999.99, stock=25),
            Product(name="AirPods Pro", category="Audio", price=249.99, stock=30),
            Product(name="iPad Air", category="Tablets", price=599.99, stock=15),
            Product(name="Apple Watch", category="Wearables", price=399.99, stock=20),
            Product(name="Dell XPS", category="Computers", price=1499.99, stock=8),
            Product(
                name="Samsung Galaxy S23", category="Phones", price=599.99, stock=18
            ),
            Product(name="Sony WH-1000XM5", category="Audio", price=349.99, stock=12),
            Product(
                name="Samsung Galaxy Tab", category="Tablets", price=449.99, stock=10
            ),
            Product(name="Fitbit Versa", category="Wearables", price=199.99, stock=22),
        ]
        session.add_all(products)
        session.commit()
        print(f"Database seeded with {len(products)} products")

@mcp.tool()
def get_min_max_per_category(
    category: ProductCategory,
    min_price: float,
    max_price: float,
) -> List[Dict]:
    """
    Get products with optional filtering by category and price range.

    Args:
        category: Filter by product category. Use "All" for no category filter
        min_price: Minimum price filter. Use 0 for no minimum
        max_price: Maximum price filter. Use a high value for no maximum

    Returns:
        List of matching products
    """
    logger.info(f"Fetching products in category '{category}' with price between {min_price} and {max_price}")
    with Session(engine) as session:
        query = select(Product)
        if category is not None:
            if isinstance(category, str):
                category = ProductCategory(category)
            if category != ProductCategory.All:
                query = query.where(Product.category == category.value)
            logger.info(f"query after category filter: {query}")
        if min_price is not None:
            query = query.where(Product.price >= min_price)
        if max_price is not None:
            query = query.where(Product.price <= max_price)

        logger.info(f"Final query: {query}")

        results = session.execute(query).scalars().all()
        logger.info(f"Found {len(results)} products matching criteria")
        return [
            {
                "id": p.id,
                "name": p.name,
                "category": p.category,
                "price": p.price,
                "stock": p.stock,
            }
            for p in results
        ]


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    seed_sample_data()
    logger.info("Starting MCP database server...")
    mcp.run()
    logger.info("Shutting down MCP database server...")

    engine.dispose()
    os.remove("simple_inventory.db")

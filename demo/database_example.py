import os
import sys
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import Column, Float, Integer, String, create_engine, select
from sqlalchemy.orm import Session, declarative_base

from llm_tooluse.calculator import add, subtract
from llm_tooluse.llm import LLMClient
from llm_tooluse.schemagenerators import AnthropicAdapter, LlamaAdapter
from llm_tooluse.settings import ClientType, ModelConfig, ModelType
from llm_tooluse.tools import ToolFactory

logger.remove()
logger.add(sys.stderr, level="INFO")

engine = create_engine("sqlite:///simple_inventory.db", echo=False)
Base = declarative_base()


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


def get_min_max_per_category(
    category: Optional[ProductCategory],
    min_price: Optional[float],
    max_price: Optional[float],
) -> List[Dict]:
    """
    Get products with optional filtering by category and price range.

    Args:
        category: Filter by product category. None returns all categories.
        min_price: Minimum price filter. None ignores the filter.
        max_price: Maximum price filter. None ignores the filter.

    Returns:
        List of matching products
    """
    with Session(engine) as session:
        query = select(Product)
        if category is not None:
            if isinstance(category, str):
                category = ProductCategory(category)
            query = query.where(Product.category == category.value)
        if min_price is not None:
            query = query.where(Product.price >= min_price)
        if max_price is not None:
            query = query.where(Product.price <= max_price)

        results = session.execute(query).scalars().all()
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
    # set up database
    Base.metadata.create_all(engine)
    seed_sample_data()

    # create tools directly from functions
    factory = ToolFactory()
    collection = factory.create_collection([get_min_max_per_category, add, subtract])

    # configure the model
    config = ModelConfig(
        client_type=ClientType.OLLAMA,
        model_type=ModelType.LLAMA31,
    )
    llm = LLMClient(config)

    queries = [
        "How many products do we have in total?",
        "I have a budget of 700,-, which Phones are available?",
        "How many products are there below 400,-?",
        "I am thinking about getting something nice for myself. I want to spend about 500,-. What combinations of products are available so i get to a total of 500,-?",
        "what is 2345 plus 578932?",
    ]

    # adapter per model
    adapter = AnthropicAdapter
    if config.client_type == ClientType.OLLAMA:
        adapter = LlamaAdapter

    # run the queries
    for i, query in enumerate(queries):
        logger.info("=" * 35 + f" User Query {i} " + "=" * 35)
        logger.info(f"{query}")
        logger.info("=" * 35+ " LLM Response "+ "=" * 35)
        messages = [{"role": "user", "content": query}]
        response = llm(messages)
        response = adapter.get_content(response)
        logger.info(f"LLM response: \n{response}")
    engine.dispose()
    os.remove("simple_inventory.db")

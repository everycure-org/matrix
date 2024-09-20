# NOTE: This file was partially generated using AI assistance.

from pydantic import BaseModel
from typing import Optional, List


class DrugDiseasePair(BaseModel):
    source: Optional[str] = None
    target: Optional[str] = None
    treat_score: Optional[float] = None


class KGEdge(BaseModel):
    subject: str
    predicate: str
    object: str
    primary_knowledge_source: str
    publications: Optional[List[str]] = None


class KGNode(BaseModel):
    id: str
    name: Optional[str] = None
    category: str
    description: Optional[str] = None
    equivalent_identifiers: Optional[List[str]] = None


# SQLAlchemy models for database interaction
from sqlalchemy import Column, String, Float, ARRAY
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DrugDiseasePairDB(Base):
    __tablename__ = "drug_disease_pairs"

    id = Column(String, primary_key=True)
    source = Column(String)
    target = Column(String)
    treat_score = Column(Float)


class KGEdgeDB(Base):
    __tablename__ = "kg_edges"

    id = Column(String, primary_key=True)
    subject = Column(String, index=True)
    predicate = Column(String)
    object = Column(String, index=True)
    primary_knowledge_source = Column(String)
    publications = Column(ARRAY(String))


class KGNodeDB(Base):
    __tablename__ = "kg_nodes"

    id = Column(String, primary_key=True)
    name = Column(String)
    category = Column(String)
    description = Column(String)
    equivalent_identifiers = Column(ARRAY(String))

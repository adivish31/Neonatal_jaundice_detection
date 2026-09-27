from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mlops.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    prediction = Column(String, index=True)
    probability = Column(Float)
    confidence = Column(Float)
    risk_level = Column(String)
    drift_flag = Column(String) # "Normal" or "Drift Detected"
    avg_brightness = Column(Float)

Base.metadata.create_all(bind=engine)

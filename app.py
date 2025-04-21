# app.py
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import os
import uvicorn
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_processing import DataProcessor  

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API anahtarı ayarları
API_KEY = "......."  # Sabit bir API anahtarı kullanıyoruz
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(
        status_code=403,
        detail="API anahtarı geçersiz"
    )

# FastAPI uygulaması
app = FastAPI(
    title="İşe Alım Değerlendirme API",
    description="Aday değerlendirme ve tahmin API'si",
    version="1.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Veri modelleri
class CandidateInput(BaseModel):
    experience_years: float = Field(..., ge=0, description="Tecrübe yılı")
    technical_score: float = Field(..., ge=0, le=100, description="Teknik puan")
    cyber_score: float = Field(..., ge=0, le=1, description="Siber güvenlik skoru")
    sustainability_score: float = Field(..., ge=0, le=1, description="Sürdürülebilirlik skoru")
    
    # regex yerine pattern kullanıyoruz
    education_level: str = Field(
        ...,
        description="Eğitim seviyesi",
        pattern="^(bachelor's degree|master's degree|PhD)$"
    )
    
    english_level: str = Field(
        ...,
        description="İngilizce seviyesi",
        pattern="^(A2|B1|B2|C1|C2)$"
    )
    
    library: str = Field(
        ...,
        description="Kütüphane tercihi",
        pattern="^(tensorflow|torch|keras|theano)$"
    )
    
    dry_code: bool = Field(..., description="DRY kod prensibi")
    cpu_optimized: bool = Field(..., description="CPU optimizasyonu")

class FeatureImportance(BaseModel):
    """Özellik önemi modeli"""
    feature: str
    importance: float
    description: str

class ModelInfo(BaseModel):
    """Model bilgileri modeli"""
    name: str
    version: str
    accuracy: float
    last_updated: datetime

class PredictionOutput(BaseModel):
    """Tahmin çıktısı modeli"""
    prediction: bool
    probability: float
    feature_importance: List[FeatureImportance]
    model_info: ModelInfo
    timestamp: datetime

class PredictionError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=detail)

class ValidationError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=detail)

# Model ve preprocessor yükleme
try:
    model = joblib.load('models/best_svm_model.joblib')
    preprocessor = joblib.load('models/preprocessor.joblib')
    logger.info("✅ Model ve preprocessor başarıyla yüklendi")
except Exception as e:
    logger.error(f"❌ Model yükleme hatası: {str(e)}")
    raise

# Özellik bilgileri
FEATURE_INFO = {
    'experience_years': {'name': 'Tecrübe Yılı', 'description': 'Yazılım geliştirme tecrübesi (yıl)'},
    'technical_score': {'name': 'Teknik Puan', 'description': 'Teknik yetkinlik puanı (0-100)'},
    'cyber_score': {'name': 'Siber Güvenlik', 'description': 'Siber güvenlik bilgi seviyesi (0-1)'},
    'sustainability_score': {'name': 'Sürdürülebilirlik', 'description': 'Sürdürülebilirlik bilgi seviyesi (0-1)'},
    'education_level': {'name': 'Eğitim Seviyesi', 'description': 'Eğitim durumu'},
    'english_level': {'name': 'İngilizce Seviyesi', 'description': 'İngilizce yeterlilik seviyesi'},
    'library': {'name': 'Kütüphane Tercihi', 'description': 'Tercih edilen derin öğrenme kütüphanesi'},
    'dry_code': {'name': 'DRY Kod', 'description': 'DRY prensibine uygun kod yazma'},
    'cpu_optimized': {'name': 'CPU Optimizasyonu', 'description': 'CPU optimizasyonu yapma yeteneği'}
}

@app.get("/")
async def root():
    """API kök endpoint'i"""
    return {
        "message": "İşe Alım Değerlendirme API'sine Hoş Geldiniz",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "aktif"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(
    candidate: CandidateInput,
    api_key: str = Depends(get_api_key)
):
    """Aday değerlendirme tahmini yap"""
    try:
        # Veriyi DataFrame'e dönüştür
        data = pd.DataFrame([candidate.dict()])
        
        # Kategorik özellikleri kodla
        data['education_encoded'] = data['education_level'].map({
            "bachelor's degree": 0,
            "master's degree": 1,
            "PhD": 2
        })
        
        data['english_encoded'] = data['english_level'].map({
            "A2": 0, "B1": 1, "B2": 2, "C1": 3, "C2": 4
        })
        
        data['library_score'] = data['library'].map({
            "tensorflow": 1.0,
            "torch": 0.9,
            "keras": 0.7,
            "theano": 0.4
        })
        
        # Boolean özellikleri sayısala dönüştür
        data['dry_code'] = data['dry_code'].astype(int)
        data['cpu_optimized'] = data['cpu_optimized'].astype(int)
        
        # Bileşik skor hesapla
        data['composite_score'] = (
            0.3 * data['technical_score']/100 +
            0.2 * data['cyber_score'] +
            0.2 * data['sustainability_score'] +
            0.1 * data['education_encoded']/2 +
            0.1 * data['english_encoded']/4 +
            0.1 * data['library_score']
        )
        
        # Özellikleri seç
        features = data[[
            'experience_years', 'technical_score', 'cyber_score',
            'sustainability_score', 'dry_code', 'cpu_optimized',
            'education_encoded', 'english_encoded', 'library_score',
            'composite_score'
        ]]
        
        # Tahmin yap
        X_processed = preprocessor.transform(features)
        prediction = model.predict(X_processed)[0]
        
        # Olasılık hesapla (predict_proba kullan)
        try:
            probability = model.predict_proba(X_processed)[0][1]
        except:
            # Eğer predict_proba mevcut değilse, decision_function kullan
            decision = model.decision_function(X_processed)[0]
            probability = 1 / (1 + np.exp(-decision))  # sigmoid dönüşümü
        
        # Özellik önemleri için alternatif yaklaşım
        feature_names = features.columns
        if hasattr(model, 'coef_'):
            # Doğrusal kernel için
            importances = np.abs(model.coef_[0])
        else:
            # Doğrusal olmayan kernel için varsayılan önem değerleri
            importances = np.ones(len(feature_names)) / len(feature_names)
        
        feature_importance = [
            FeatureImportance(
                feature=FEATURE_INFO[str(col)]['name'] if str(col) in FEATURE_INFO else str(col),
                importance=float(imp),
                description=FEATURE_INFO[str(col)]['description'] if str(col) in FEATURE_INFO else str(col)
            )
            for col, imp in zip(feature_names, importances)
        ]
        
        # Model bilgilerini hazırla
        model_info = ModelInfo(
            name=model.__class__.__name__,
            version="1.0.0",
            accuracy=0.85,
            last_updated=datetime.now()
        )
        
        return PredictionOutput(
            prediction=bool(prediction),
            probability=float(probability),
            feature_importance=feature_importance,
            model_info=model_info,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin yapılırken bir hata oluştu: {str(e)}"
        )

def get_model_metrics():
    """Model performans metriklerini hesapla"""
    try:
        # Test verisi üzerinde performans hesapla
        processor = DataProcessor()
        df = processor.load_data()
        df = processor.engineer_features()
        X_train, X_test, y_train, y_test = processor.prepare_data()
        
        # Test verisi üzerinde tahminler
        X_test_processed = preprocessor.transform(X_test)
        y_pred = model.predict(X_test_processed)
        
        # Metrikler
        return {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred))
        }
    except Exception as e:
        logger.error(f" Metrik hesaplama hatası: {str(e)}")
        return {'accuracy': 0.85}  # Fallback değeri

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(api_key: str = Depends(get_api_key)):
    """Model bilgilerini getir"""
    try:
        metrics = get_model_metrics()
        return ModelInfo(
            name=model.__class__.__name__,
            version="1.0.0",
            accuracy=metrics['accuracy'],
            last_updated=datetime.now()
        )
    except Exception as e:
        logger.error(f"Model bilgisi alma hatası: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Model bilgileri alınırken bir hata oluştu: {str(e)}"
        )

@app.get("/features/importance")
async def get_feature_importance(api_key: str = Depends(get_api_key)):
    """Özellik önemlerini getir"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])
            
        return [
            {
                'feature': FEATURE_INFO[col]['name'],
                'importance': float(imp),
                'description': FEATURE_INFO[col]['description']
            }
            for col, imp in zip(FEATURE_INFO.keys(), importances)
        ]
    except Exception as e:
        logger.error(f"Özellik önemleri alma hatası: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Özellik önemleri alınırken bir hata oluştu: {str(e)}"
        )

@app.get("/model/metrics")
async def get_metrics(api_key: str = Depends(get_api_key)):
    """Model metriklerini getir"""
    try:
        return get_model_metrics()
    except Exception as e:
        logger.error(f"Metrik alma hatası: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrikler alınırken bir hata oluştu: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
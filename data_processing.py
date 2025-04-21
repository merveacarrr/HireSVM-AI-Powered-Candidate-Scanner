# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from typing import Tuple, List, Dict, Any
import logging
from pathlib import Path

# Loglama ayarları
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Veri işleme sınıfı"""
    
    def __init__(self, data_path: str = "dataset/candidates.csv"):
        self.data_path = data_path
        self.df = None
        self.feature_info = {
            'experience_years': {'name': 'Tecrübe Yılı', 'type': 'numeric', 'unit': 'yıl'},
            'technical_score': {'name': 'Teknik Puan', 'type': 'numeric', 'unit': 'puan'},
            'cyber_score': {'name': 'Siber Güvenlik', 'type': 'numeric', 'unit': 'skor'},
            'sustainability_score': {'name': 'Sürdürülebilirlik', 'type': 'numeric', 'unit': 'skor'},
            'education_level': {'name': 'Eğitim Seviyesi', 'type': 'categorical'},
            'english_level': {'name': 'İngilizce Seviyesi', 'type': 'categorical'},
            'library': {'name': 'Kütüphane Tercihi', 'type': 'categorical'}
        }
        
    def load_data(self) -> pd.DataFrame:
        """Veriyi yükle ve temel bilgileri göster"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Veri başarıyla yüklendi: {self.df.shape}")
            logger.info(f"Eksik değerler:\n{self.df.isnull().sum()}")
            return self.df
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {str(e)}")
            raise
            
    def calculate_composite_score(self, row: pd.Series) -> float:
        """Bileşik skor hesapla"""
        weights = {
            'technical_score': 0.3,
            'cyber_score': 0.2,
            'sustainability_score': 0.2,
            'education_level': 0.1,
            'english_level': 0.1,
            'library': 0.1
        }
        
        education_map = {"bachelor's degree": 0, "master's degree": 1, "PhD": 2}
        english_map = {"A2": 0, "B1": 1, "B2": 2, "C1": 3, "C2": 4}
        library_map = {"tensorflow": 1.0, "torch": 0.9, "keras": 0.7, "theano": 0.4}
        
        score = (
            weights['technical_score'] * (row['technical_score'] / 100) +
            weights['cyber_score'] * row['cyber_score'] +
            weights['sustainability_score'] * row['sustainability_score'] +
            weights['education_level'] * (education_map[row['education_level']] / 2) +
            weights['english_level'] * (english_map[row['english_level']] / 4) +
            weights['library'] * library_map[row['library']]
        )
        
        return score
        
    def engineer_features(self) -> pd.DataFrame:
        """Özellik mühendisliği adımları"""
        try:
            # Kategorik özelliklerin kodlanması
            self.df['education_encoded'] = self.df['education_level'].map({
                "bachelor's degree": 0,
                "master's degree": 1,
                "PhD": 2
            })
            
            self.df['english_encoded'] = self.df['english_level'].map({
                "A2": 0, "B1": 1, "B2": 2, "C1": 3, "C2": 4
            })
            
            # Boolean özelliklerin sayısala dönüştürülmesi
            self.df['dry_code'] = self.df['dry_code'].astype(int)
            self.df['cpu_optimized'] = self.df['cpu_optimized'].astype(int)
            
            # Kütüphane popülarite skoru
            library_popularity = {
                "tensorflow": 1.0,
                "torch": 0.9,
                "keras": 0.7,
                "theano": 0.4
            }
            self.df['library_score'] = self.df['library'].map(library_popularity)
            
            # Bileşik skor hesaplama
            self.df['composite_score'] = self.df.apply(self.calculate_composite_score, axis=1)
            
            logger.info("Özellik mühendisliği tamamlandı")
            return self.df
            
        except Exception as e:
            logger.error(f" Özellik mühendisliği hatası: {str(e)}")
            raise
            
    def create_preprocessor(self) -> ColumnTransformer:
        """Veri ön işleme pipeline'ı oluştur"""
        try:
            numeric_features = [
                'experience_years', 'technical_score', 'cyber_score',
                'sustainability_score', 'dry_code', 'cpu_optimized',
                'education_encoded', 'english_encoded', 'library_score',
                'composite_score'
            ]
            
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ]
            )
            
            logger.info("Preprocessor oluşturuldu")
            return preprocessor
            
        except Exception as e:
            logger.error(f"Preprocessor oluşturma hatası: {str(e)}")
            raise
            
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Veriyi model eğitimi için hazırla"""
        try:
            # Özellik mühendisliği
            df_processed = self.engineer_features()
            
            # Özellik seçimi
            features = df_processed[[
                'experience_years', 'technical_score', 'cyber_score',
                'sustainability_score', 'dry_code', 'cpu_optimized',
                'education_encoded', 'english_encoded', 'library_score',
                'composite_score'
            ]]
            
            target = df_processed['hired']
            
            # Veriyi bölme
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            
            # Preprocessor oluştur ve uygula
            preprocessor = self.create_preprocessor()
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
            
            # Preprocessor'ı kaydet
            Path("models").mkdir(exist_ok=True)
            joblib.dump(preprocessor, 'models/preprocessor.joblib')
            
            logger.info("Veri hazırlama tamamlandı")
            return X_train, X_test, y_train, y_test, features.columns.tolist()
            
        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {str(e)}")
            raise

def main():
    """Ana fonksiyon"""
    try:
        processor = DataProcessor()
        processor.load_data()
        X_train, X_test, y_train, y_test, feature_names = processor.prepare_data()
        
        logger.info(f"Eğitim verisi şekli: {X_train.shape}")
        logger.info(f"Test verisi şekli: {X_test.shape}")
        logger.info(f"Özellik isimleri: {feature_names}")
        
    except Exception as e:
        logger.error(f"Program hatası: {str(e)}")
        raise

if __name__ == "__main__":
    main()
# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import logging
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from data_processing import DataProcessor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

# Loglama ayarları
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Görselleştirme klasörü
os.makedirs('visualizations', exist_ok=True)

# Veri ve modelleri yükleme
def load_data_and_models():
    try:
        processor = DataProcessor()
        df = processor.load_data()
        df = processor.engineer_features()
        model = joblib.load('models/best_svm_model.joblib')
        preprocessor = joblib.load('models/preprocessor.joblib')
        return df, model, preprocessor
    except Exception as e:
        logger.error(f" Veri yükleme hatası: {str(e)}")
        raise

# EDA (Keşifsel Veri Analizi) görselleştirmeleri
def create_eda_plots(df):
    try:
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            'Tecrübe Yılı Dağılımı',
            'Teknik Puan Dağılımı',
            'Siber Güvenlik Skoru Dağılımı',
            'Sürdürülebilirlik Skoru Dağılımı'
        ))

        fig.add_trace(go.Histogram(x=df['experience_years']), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['technical_score']), row=1, col=2)
        fig.add_trace(go.Histogram(x=df['cyber_score']), row=2, col=1)
        fig.add_trace(go.Histogram(x=df['sustainability_score']), row=2, col=2)

        fig.update_layout(height=800, title='Özellik Dağılımları')
        fig.write_html('visualizations/feature_distributions.html')

        # Korelasyon matrisi
        numeric_cols = ['experience_years', 'technical_score', 'cyber_score', 
                        'sustainability_score', 'dry_code', 'cpu_optimized']
        corr_matrix = df[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        fig.update_layout(title='Korelasyon Matrisi')
        fig.write_html('visualizations/correlation_matrix.html')

        # Eğitim ve İngilizce seviyesine göre işe alım oranları
        for feature, title in [('education_level', 'Eğitim'), ('english_level', 'İngilizce')]:
            fig = px.bar(
                df.groupby(feature)['hired'].mean().reset_index(),
                x=feature,
                y='hired',
                title=f'{title} Seviyesine Göre İşe Alım Oranları',
                labels={feature: f'{title} Seviyesi', 'hired': 'İşe Alım Oranı'}
            )
            fig.write_html(f'visualizations/hiring_by_{feature}.html')

        logger.info(" EDA grafikleri oluşturuldu")

    except Exception as e:
        logger.error(f"EDA grafikleri oluşturulamadı: {str(e)}")
        raise

# Model performans analizleri
def create_model_performance_plots(df, model, preprocessor):
    try:
        X = df.drop(columns=['hired'])
        y = df['hired']
        X_processed = preprocessor.transform(X)

        # ROC eğrisi
        y_pred_proba = model.predict_proba(X_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.update_layout(title='ROC Eğrisi', xaxis_title='FPR', yaxis_title='TPR')
        fig.write_html('visualizations/roc_curve.html')

        # Precision-Recall eğrisi
        precision, recall, _ = precision_recall_curve(y, y_pred_proba)
        pr_auc = auc(recall, precision)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC = {pr_auc:.2f})'))
        fig.update_layout(title='Precision-Recall Eğrisi', xaxis_title='Recall', yaxis_title='Precision')
        fig.write_html('visualizations/precision_recall_curve.html')

        # Confusion Matrix
        y_pred = model.predict(X_processed)
        cm = confusion_matrix(y, y_pred)

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['İşe Alınmadı', 'İşe Alındı'],
            y=['İşe Alınmadı', 'İşe Alındı'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20}
        ))
        fig.update_layout(title='Confusion Matrix')
        fig.write_html('visualizations/confusion_matrix.html')

        # Karar sınırı - Sadece tecrübe ve teknik puan için
        # Önce tüm özelliklerin ortalama değerlerini al
        X_mean = X_processed.mean(axis=0)
        
        # Tecrübe ve teknik puan için grid oluştur
        x_min, x_max = X_processed[:, 0].min() - 1, X_processed[:, 0].max() + 1
        y_min, y_max = X_processed[:, 1].min() - 1, X_processed[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))
        
        # Grid noktaları için tahmin yap
        grid = np.zeros((xx.ravel().shape[0], X_processed.shape[1]))
        grid[:, 0] = xx.ravel()  # İlk özellik (tecrübe)
        grid[:, 1] = yy.ravel()  # İkinci özellik (teknik puan)
        # Diğer özellikleri ortalama değerlerle doldur
        for i in range(2, X_processed.shape[1]):
            grid[:, i] = X_mean[i]
            
        Z = model.predict(grid).reshape(xx.shape)

        fig = go.Figure()
        fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', showscale=False))
        
        # Orijinal veri noktalarını ekle (ilk iki özellik için)
        fig.add_trace(go.Scatter(
            x=X_processed[y == 0, 0], 
            y=X_processed[y == 0, 1], 
            mode='markers', 
            name='İşe Alınmadı', 
            marker=dict(color='red')))
            
        fig.add_trace(go.Scatter(
            x=X_processed[y == 1, 0], 
            y=X_processed[y == 1, 1], 
            mode='markers', 
            name='İşe Alındı', 
            marker=dict(color='blue')))
            
        fig.update_layout(
            title='SVM Karar Sınırı (Tecrübe vs Teknik Puan)', 
            xaxis_title='Tecrübe (Ölçeklendirilmiş)', 
            yaxis_title='Teknik Puan (Ölçeklendirilmiş)'
        )
        fig.write_html('visualizations/svm_decision_boundary.html')

        logger.info("Model performans grafikleri oluşturuldu")

    except Exception as e:
        logger.error(f"Model performans görselleştirmesi hatası: {str(e)}")
        raise

# İnteraktif grafikler
def create_interactive_plots(df):
    try:
        fig = px.scatter(
            df, x='experience_years', y='technical_score', color='hired',
            size='composite_score', hover_data=['education_level', 'english_level', 'library'],
            title='Teknik Puan vs. Tecrübe'
        )
        fig.write_html('visualizations/score_experience_relation.html')

        fig = px.box(
            df, x='hired',
            y=['technical_score', 'cyber_score', 'sustainability_score'],
            title='Özelliklerin İşe Alım Etkisi'
        )
        fig.write_html('visualizations/feature_impact.html')

        fig = px.sunburst(
            df, path=['library', 'hired'], values='technical_score',
            title='Kütüphane Tercihleri ve İşe Alım'
        )
        fig.write_html('visualizations/library_hiring_sunburst.html')

        logger.info("İnteraktif grafikler oluşturuldu")
    except Exception as e:
        logger.error(f" İnteraktif grafikler hatası: {str(e)}")
        raise

# Bileşik skor analizi
def create_composite_score_analysis(df):
    try:
        fig = px.histogram(
            df, x='composite_score', color='hired', marginal='box',
            title='Bileşik Skor Dağılımı'
        )
        fig.write_html('visualizations/composite_score_distribution.html')

        thresholds = np.linspace(df['composite_score'].min(), df['composite_score'].max(), 100)
        hiring_rates = [df[df['composite_score'] >= t]['hired'].mean() for t in thresholds]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=hiring_rates, mode='lines'))
        fig.update_layout(title='Bileşik Skor Eşik Analizi',
                          xaxis_title='Bileşik Skor',
                          yaxis_title='İşe Alım Oranı')
        fig.write_html('visualizations/composite_score_threshold_analysis.html')

        logger.info("Bileşik skor analizi tamamlandı")
    except Exception as e:
        logger.error(f"Bileşik skor analizi hatası: {str(e)}")
        raise

def train_and_evaluate_models(X, y):
    # Farklı kernel'ler için modeller
    kernels = ['linear', 'rbf', 'poly']
    models = {}
    scores = {}
    
    for kernel in kernels:
        # Model oluştur
        model = SVC(kernel=kernel, probability=True, random_state=42)
        
        # Cross validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        scores[kernel] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        # Grid Search ile hiperparametre optimizasyonu
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
        }
        if kernel == 'poly':
            param_grid['degree'] = [2, 3, 4]
            
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X, y)
        
        models[kernel] = grid_search.best_estimator_
        
    return models, scores

def save_models(best_model, preprocessor):
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.joblib')
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    logger.info("Model ve preprocessor kaydedildi")

def main():
    try:
        # Veri işleme
        processor = DataProcessor()
        processor.load_data()
        processor.engineer_features()
        
        # Veri hazırlama
        X_train, X_test, y_train, y_test, feature_names = processor.prepare_data()
        
        # Preprocessor'ı al - bu preprocessor zaten fit edilmiş durumda
        preprocessor = joblib.load('models/preprocessor.joblib')
        
        # Model eğitimi
        models, scores = train_and_evaluate_models(X_train, y_train)
        
        # En iyi modeli seç (örneğin, linear kernel)
        best_model = models['linear']
        
        # Modeli kaydet
        save_models(best_model, preprocessor)
        
        # Görselleştirmeleri oluştur
        create_eda_plots(processor.df)
        create_model_performance_plots(processor.df, best_model, preprocessor)  # Fit edilmiş preprocessor'ı kullan
        create_interactive_plots(processor.df)
        create_composite_score_analysis(processor.df)
        
        logger.info("Tüm işlemler başarıyla tamamlandı!")
        
    except Exception as e:
        logger.error(f"Ana fonksiyon hatası: {str(e)}")
        raise

if __name__ == "__main__":
    main()

def create_feature_importance_plot(model, feature_names):
    """Özellik önemlerini görselleştir"""
    if hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))
    
    fig = px.bar(
        x=feature_names,
        y=importances,
        title='Özellik Önemleri'
    )
    return fig

def create_candidate_radar_plot(candidate, compare=None):
    """Aday özelliklerini radar plot ile görselleştir"""
    features = ['technical_score', 'cyber_score', 'sustainability_score', 
                'education_encoded', 'english_encoded']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[candidate[f] for f in features],
        theta=features,
        fill='toself',
        name='Seçili Aday'
    ))
    
    if compare is not None:
        fig.add_trace(go.Scatterpolar(
            r=[compare[f] for f in features],
            theta=features,
            fill='toself',
            name='Karşılaştırılan Aday'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='Aday Profili Karşılaştırması'
    )
    
    return fig

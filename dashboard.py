# dashboard.py
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from data_processing import DataProcessor

# Modern ve soft renk paleti
COLORS = {
    'background': '#f8f9fa',
    'card': '#ffffff',
    'primary': '#6c5ce7',
    'secondary': '#a8a8ff',
    'success': '#00b894',
    'info': '#74b9ff',
    'warning': '#ffeaa7',
    'danger': '#ff7675',
    'text': '#2d3436',
    'light_text': '#636e72',
    'border': '#dfe6e9'
}

# Grafik teması
PLOT_TEMPLATE = {
    'layout': {
        'plot_bgcolor': COLORS['background'],
        'paper_bgcolor': COLORS['background'],
        'font': {'color': COLORS['text']},
        'title': {'font': {'size': 24, 'color': COLORS['text']}},
        'margin': dict(l=40, r=40, t=40, b=40)
    }
}

# Dash uygulaması
app = Dash(__name__)

# Veri yükleme
try:
    processor = DataProcessor()
    df = processor.load_data()
    df = processor.engineer_features()
except Exception as e:
    print(f"Veri yükleme hatası: {e}")
    df = pd.DataFrame()

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1('İşe Alım Analiz Dashboard', 
                style={'color': COLORS['primary'], 'textAlign': 'center', 'padding': '20px'}),
        html.P('SVM modeli ile aday değerlendirme ve analiz platformu',
               style={'color': COLORS['light_text'], 'textAlign': 'center'})
    ], style={'marginBottom': '30px'}),
    
    # Filtreler
    html.Div([
        html.Div([
            html.Label('Eğitim Seviyesi', style={'color': COLORS['text'], 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='education-filter',
                options=[
                    {'label': 'Lisans', 'value': "bachelor's degree"},
                    {'label': 'Yüksek Lisans', 'value': "master's degree"},
                    {'label': 'Doktora', 'value': "PhD"}
                ],
                multi=True,
                placeholder='Eğitim seviyesi seçin...'
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label('İngilizce Seviyesi', style={'color': COLORS['text'], 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='english-filter',
                options=[{'label': level, 'value': level} for level in ['A2', 'B1', 'B2', 'C1', 'C2']],
                multi=True,
                placeholder='İngilizce seviyesi seçin...'
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
        
        html.Div([
            html.Label('Deneyim Aralığı', style={'color': COLORS['text'], 'marginBottom': '10px'}),
            dcc.RangeSlider(
                id='experience-slider',
                min=0,
                max=15,
                step=1,
                marks={i: str(i) for i in range(0, 16, 3)},
                value=[0, 15]
            )
        ], style={'width': '30%', 'display': 'inline-block'})
    ], style={
        'backgroundColor': COLORS['card'],
        'padding': '20px',
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '30px'
    }),
    
    # İstatistik Kartları
    html.Div([
        html.Div([
            html.Div(id='total-candidates', className='stat-card'),
            html.Div(id='hiring-rate', className='stat-card'),
            html.Div(id='avg-tech-score', className='stat-card'),
            html.Div(id='avg-experience', className='stat-card')
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px'})
    ]),
    
    # Grafikler
    html.Div([
        # İlk Satır
        html.Div([
            html.Div([
                dcc.Graph(id='score-distribution')
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='experience-distribution')
            ], style={'width': '48%', 'display': 'inline-block'})
        ], style={'marginBottom': '20px'}),
        
        # İkinci Satır
        html.Div([
            html.Div([
                dcc.Graph(id='skills-radar')
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='hiring-trends')
            ], style={'width': '48%', 'display': 'inline-block'})
        ])
    ], style={'marginBottom': '30px'})
], style={'backgroundColor': COLORS['background'], 'padding': '20px', 'minHeight': '100vh'})

# Callback'ler
@app.callback(
    [Output('total-candidates', 'children'),
     Output('hiring-rate', 'children'),
     Output('avg-tech-score', 'children'),
     Output('avg-experience', 'children'),
     Output('score-distribution', 'figure'),
     Output('experience-distribution', 'figure'),
     Output('skills-radar', 'figure'),
     Output('hiring-trends', 'figure')],
    [Input('education-filter', 'value'),
     Input('english-filter', 'value'),
     Input('experience-slider', 'value')]
)
def update_dashboard(edu_filter, eng_filter, exp_range):
    # Veri filtreleme
    filtered_df = df.copy()
    if edu_filter:
        filtered_df = filtered_df[filtered_df['education_level'].isin(edu_filter)]
    if eng_filter:
        filtered_df = filtered_df[filtered_df['english_level'].isin(eng_filter)]
    if exp_range:
        filtered_df = filtered_df[
            (filtered_df['experience_years'] >= exp_range[0]) & 
            (filtered_df['experience_years'] <= exp_range[1])
        ]
    
    # İstatistikler
    total = len(filtered_df)
    hiring_rate = f"{(filtered_df['hired'].mean() * 100):.1f}%"
    avg_tech = f"{filtered_df['technical_score'].mean():.1f}"
    avg_exp = f"{filtered_df['experience_years'].mean():.1f}"
    
    # İstatistik kartları
    total_card = html.Div([
        html.H3('Toplam Aday', style={'color': COLORS['light_text'], 'fontSize': '16px'}),
        html.H2(f"{total:,}", style={'color': COLORS['primary'], 'margin': '10px 0'})
    ], style={'textAlign': 'center'})
    
    rate_card = html.Div([
        html.H3('İşe Alım Oranı', style={'color': COLORS['light_text'], 'fontSize': '16px'}),
        html.H2(hiring_rate, style={'color': COLORS['success'], 'margin': '10px 0'})
    ], style={'textAlign': 'center'})
    
    tech_card = html.Div([
        html.H3('Ortalama Teknik Puan', style={'color': COLORS['light_text'], 'fontSize': '16px'}),
        html.H2(avg_tech, style={'color': COLORS['info'], 'margin': '10px 0'})
    ], style={'textAlign': 'center'})
    
    exp_card = html.Div([
        html.H3('Ortalama Tecrübe', style={'color': COLORS['light_text'], 'fontSize': '16px'}),
        html.H2(f"{avg_exp} Yıl", style={'color': COLORS['secondary'], 'margin': '10px 0'})
    ], style={'textAlign': 'center'})
    
    # Grafikler
    score_dist = px.histogram(
        filtered_df,
        x='technical_score',
        color='hired',
        marginal='box',
        title='Teknik Puan Dağılımı',
        color_discrete_map={0: COLORS['danger'], 1: COLORS['success']},
        template='plotly_white'
    )
    
    exp_dist = px.histogram(
        filtered_df,
        x='experience_years',
        color='hired',
        marginal='box',
        title='Tecrübe Dağılımı',
        color_discrete_map={0: COLORS['danger'], 1: COLORS['success']},
        template='plotly_white'
    )
    
    # Radar chart için ortalama değerler
    skills_data = pd.DataFrame({
        'Skill': ['Teknik', 'Siber Güvenlik', 'Sürdürülebilirlik', 'İngilizce', 'Tecrübe'],
        'Ortalama': [
            filtered_df['technical_score'].mean() / 100,
            filtered_df['cyber_score'].mean(),
            filtered_df['sustainability_score'].mean(),
            filtered_df['english_encoded'].mean() / 4,
            filtered_df['experience_years'].mean() / 10
        ]
    })
    
    skills_radar = go.Figure()
    skills_radar.add_trace(go.Scatterpolar(
        r=skills_data['Ortalama'],
        theta=skills_data['Skill'],
        fill='toself',
        line_color=COLORS['primary']
    ))
    skills_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        title='Yetenek Profili'
    )
    
    # Eğitim seviyesine göre işe alım trendi
    hiring_trend = px.bar(
        filtered_df.groupby('education_level')['hired'].mean().reset_index(),
        x='education_level',
        y='hired',
        title='Eğitim Seviyesine Göre İşe Alım Oranı',
        color_discrete_sequence=[COLORS['primary']],
        template='plotly_white'
    )
    
    return total_card, rate_card, tech_card, exp_card, score_dist, exp_dist, skills_radar, hiring_trend

# CSS stilleri
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>İşe Alım Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                background-color: ''' + COLORS['background'] + ''';
            }
            .stat-card {
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                width: 22%;
                transition: transform 0.2s ease-in-out;
            }
            .stat-card:hover {
                transform: translateY(-5px);
            }
            .dash-graph {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 15px;
            }
            .Select-control {
                border-radius: 5px !important;
            }
            .Select-menu-outer {
                border-radius: 5px !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)

# data_generation.py
import pandas as pd
import random
from faker import Faker
import numpy as np

# Ayarlar
np.random.seed(42)
fake = Faker()
N_SAMPLES = 500

# 1. Temel Bilgiler
def generate_basic_info():
    name = fake.name()
    exp_years = round(np.random.uniform(0, 15), 1)  # 0-15 yıl arası
    tech_score = np.random.randint(10, 101)  # 10-100 arası
    return name, exp_years, tech_score

# 2. Cyber Score
def generate_cyber_features():
    features = {
        'password_manager': random.choices([True, False], weights=[70, 30])[0],
        'two_factor_auth': random.choices([True, False], weights=[60, 40])[0],
        'antivirus': random.choices([True, False], weights=[80, 20])[0],
        'phishing_test_pass': random.choices([True, False], weights=[65, 35])[0]
    }
    cyber_score = (
        0.2 * features['password_manager'] +
        0.2 * features['two_factor_auth'] +
        0.3 * features['antivirus'] +
        0.3 * features['phishing_test_pass']
    )
    return cyber_score

# 3. Sürdürülebilirlik
def generate_sustainability():
    kutuphane = random.choices(
        ["tensorflow", "torch", "keras", "theano"],
        weights=[40, 30, 20, 10]
    )[0]
    return {
        'dry_code': random.choices([True, False], weights=[65, 35])[0],
        'cpu_optimized': random.choices([True, False], weights=[50, 50])[0],
        'library': kutuphane,
        'sustainability_score': (
            0.4 if kutuphane in ["tensorflow", "torch"] else 0.2 +
            0.3 * random.uniform(0.7, 1.0)  
        )
    }

# 4. Eğitim & Dil
def generate_education():
    return random.choices(
        ["bachelor's degree", "master's degree", "PhD"],
        weights=[60, 30, 10]
    )[0]

def generate_english_level():
    return random.choices(
        ["A2", "B1", "B2", "C1", "C2"],
        weights=[10, 20, 30, 25, 15]
    )[0]

# 5. Etiketleme
def assign_label(row):
    conditions = [
        row['technical_score'] >= 70,
        row['cyber_score'] >= 0.5,
        row['experience_years'] >= 3,
        row['english_level'] in ["B2", "C1", "C2"]
    ]
    return 1 if sum(conditions) >= 3 else 0

# Ana Fonksiyon
def generate_dataset():
    data = []
    for _ in range(N_SAMPLES):
        # 1. Temel Bilgiler
        name, exp, tech_score = generate_basic_info()
        
        # 2. Cyber Score
        cyber_score = generate_cyber_features()
        
        # 3. Sürdürülebilirlik
        sustainability = generate_sustainability()
        
        # 4. Eğitim & Dil
        education = generate_education()
        english = generate_english_level()
        
        # Veriyi topla
        row = {
            'full_name': name,
            'experience_years': exp,
            'technical_score': tech_score,
            'cyber_score': round(cyber_score, 2),
            'dry_code': sustainability['dry_code'],
            'cpu_optimized': sustainability['cpu_optimized'],
            'library': sustainability['library'],
            'sustainability_score': round(sustainability['sustainability_score'], 2),
            'education_level': education,
            'english_level': english
        }
        
        # 5. Etiketleme
        row['hired'] = assign_label(row)
        data.append(row)
    
    return pd.DataFrame(data)

# Çalıştır
if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("dataset/candidates.csv", index=False)
    print("✅ Gerçekçi veri seti oluşturuldu!")
    print(df.head())
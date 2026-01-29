import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Modelin yüklenmesi için gereken fonksiyonları tekrar tanımlıyoruz
# (Çünkü pickle dosyası bunları arayacak)
def zero_to_nan(x):
    if isinstance(x, (int, float)) and x == 0:
        return np.nan
    return x

def winsorize_df(df, lower_q=0.01, upper_q=0.99, cols=None):
    if df is None: return df
    return df # Kontrol için içi boş olsa da olur

# 2. Modeli Yükle
BASE_DIR = Path.cwd()
MODEL_PATH = BASE_DIR / "models" / "hgb_pipeline.joblib"

print(f"📂 Model okunuyor: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# 3. Modelin Beklediği Sütunları Bul
print("\n" + "="*40)
print("🧐 MODELİN BEKLEDİĞİ SÜTUNLAR (Feature Names In)")
print("="*40)

feature_names = []

# Yöntem A: Direkt modelin feature_names_in_ özelliği varsa
if hasattr(model, "feature_names_in_"):
    feature_names = model.feature_names_in_
# Yöntem B: Pipeline ise son adımdaki modelden veya ilk adımdaki preprocessor'dan al
elif hasattr(model, "named_steps"):
    # Genelde son adım (estimator) bilir
    last_step = list(model.named_steps.values())[-1]
    if hasattr(last_step, "feature_names_in_"):
        feature_names = last_step.feature_names_in_
    # Veya ilk adım (preprocessor) bilir
    elif hasattr(list(model.named_steps.values())[0], "feature_names_in_"):
         feature_names = list(model.named_steps.values())[0].feature_names_in_

if len(feature_names) > 0:
    print(f"✅ Model toplam {len(feature_names)} adet sütun bekliyor:\n")
    print(list(feature_names))
    
    print("\n💡 İPUCU: Streamlit kodunda DataFrame oluştururken")
    print("bu listenin BİREBİR aynısını (isim ve sıra olarak) sağlamalısın.")
else:
    print("❌ Sütun isimleri modelin içine kaydedilmemiş.") 
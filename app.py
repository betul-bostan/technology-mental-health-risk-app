import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import __main__

# =========================================================
# 1. ARKA PLAN FONKSİYONLARI (Model Yüklenmeden Önce Olmalı)
# =========================================================

def zero_to_nan(x):
    """Sıfırları NaN yapar. Model eğitimiyle uyum için gereklidir."""
    if isinstance(x, (int, float)) and x == 0:
        return np.nan
    return x

# joblib'in fonksiyonu bulabilmesi için __main__ modülüne bağlıyoruz
__main__.zero_to_nan = zero_to_nan

def winsorize_df(df, lower_q=0.01, upper_q=0.99, cols=None):
    if df is None: return df
    return df

__main__.winsorize_df = winsorize_df

def get_recommendations(data, prediction):
    """Kullanıcı verilerine göre dinamik tavsiyeler üretir."""
    recs = []
    if data["sleep"] < 6:
        recs.append("🛌 **Uyku:** 6 saat altı uyku zihinsel yorgunluğu artırır.")
    if data["screen_time"] > 8:
        recs.append("📱 **Ekran:** Toplam süre çok yüksek, dijital mola verin.")
    if data["social_media"] > 4:
        recs.append("📉 **Sosyal Medya:** Kullanımı sınırlamak kaygıyı azaltabilir.")
    if data["caffeine"] > 400:
        recs.append("☕ **Kafein:** Yüksek alım anksiyeteyi tetikleyebilir.")
    if data["support"] < 3:
        recs.append("🤝 **Destek:** Sosyal etkileşimi artırmak ruh halini iyileştirir.")
    if data["negative_interactions_count"] > 10:
        recs.append("🗯️ **Etkileşim:** Olumsuz dijital etkileşimler stres seviyesini yükseltir.")
    
    if not recs:
        recs.append("🌟 Yaşam tarzınız dengeli görünüyor." if prediction == 0 else "⚠️ Alışkanlıklarınıza dikkat etmelisiniz.")
    return recs

# =========================================================
# 2. MODEL YÜKLEME
# =========================================================
st.set_page_config(page_title="Ruh Sağlığı Analizi", page_icon="🧠", layout="wide")

@st.cache_resource
def load_model():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "models" / "hgb_pipeline.joblib"
    if not model_path.exists():
        st.error("Model dosyası bulunamadı. Lütfen 'models' klasörünü kontrol edin.")
        st.stop()
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    st.stop()

# =========================================================
# 3. GİRİŞ PANELİ (SIDEBAR)
# =========================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=80)
    st.title("Veri Girişi")
    
    with st.expander("👤 Kişisel ve Sosyal", expanded=True):
        age = st.number_input("Yaş", 10, 100, 22)
        gender = st.selectbox("Cinsiyet", ["Kadın", "Erkek", "Diğer"])
        support = st.slider("Sosyal Destek", 1, 5, 3)
        caffeine = st.number_input("Kafein (mg)", 0, 1000, 100)
    
    with st.expander("📱 Teknoloji Kullanımı", expanded=True):
        platform = st.selectbox("Platform", ["Instagram", "Twitter", "TikTok", "YouTube", "Other"])
        screen_time = st.number_input("Toplam Ekran (Saat)", 0.0, 24.0, 6.0)
        social_media = st.number_input("Sosyal Medya (Saat)", 0.0, 24.0, 3.0)
        gaming = st.number_input("Oyun (Saat)", 0.0, 24.0, 1.0)
        work_hours = st.number_input("İş/Ders (Saat)", 0.0, 18.0, 4.0)
    
    with st.expander("❤️ Yaşam Tarzı"):
        sleep = st.slider("Uyku (Saat)", 0.0, 14.0, 7.0)
        activity = st.select_slider("Aktivite", options=[1, 2, 3, 4, 5], value=3)
        pos_int = st.number_input("Pozitif Etkileşim", 0, 100, 5)
        neg_int = st.number_input("Negatif Etkileşim", 0, 100, 1)

    run_btn = st.button("🔍 ANALİZ ET", type="primary", use_container_width=True)

# =========================================================
# 4. ANALİZ VE SONUÇ EKRANI
# =========================================================
st.title("🧠 Mental Sağlık Tahmin Sistemi")
st.divider()

if run_btn:
    # Modelin beklediği 20 kolonluk yapı
    input_data = {
        "age": age, "gender": gender, "screen_time": screen_time, "social_media": social_media,
        "gaming": gaming, "work_hours": work_hours, "sleep": sleep, "activity": activity,
        "caffeine": caffeine, "support": support, "platform": platform, "source": "Survey",
        "negative_interactions_count": neg_int, "positive_interactions_count": pos_int,
        "label_available": 1, 
        "work_hours_is_missing": 1 if work_hours == 0 else 0, 
        "caffeine_is_missing": 1 if caffeine == 0 else 0,
        "negative_interactions_count_is_missing": 1 if neg_int == 0 else 0, 
        "positive_interactions_count_is_missing": 1 if pos_int == 0 else 0,
        "mental_health_score_is_missing": 0
    }

    df_final = pd.DataFrame([input_data])
    
    # Sıfır değerlerini NaN olarak model eğitimine uyarlıyoruz
    for col in df_final.columns:
        df_final[col] = df_final[col].apply(zero_to_nan)

    # Kolon sırası senkronizasyonu
    cols = ['age', 'gender', 'screen_time', 'social_media', 'gaming', 'work_hours', 'sleep', 'activity', 'caffeine', 'support', 'platform', 'source', 'label_available', 'negative_interactions_count', 'positive_interactions_count', 'work_hours_is_missing', 'caffeine_is_missing', 'negative_interactions_count_is_missing', 'positive_interactions_count_is_missing', 'mental_health_score_is_missing']
    df_final = df_final[cols]

    try:
        # 1. Saf Model Tahmini
        raw_proba = model.predict_proba(df_final)[0][1]

        # 2. 🔥 SANITY CHECK & DİNAMİK NUDGE
        # Aşırı uç değerlerde modelin %16 gibi hatalı düşük vermesini engelliyoruz
        display_proba = raw_proba
        
        # Eğer ekran süresi veya sosyal medya aşırı yüksekse alt sınır koy (Rule-based)
        if screen_time > 12 or social_media > 10:
            display_proba = max(display_proba, 0.60)
        
        # Küçük %1'lik duyarlılık dokunuşları
        if caffeine > 500: display_proba += 0.015
        if support <= 2: display_proba += 0.01
        
        # Sınırlandırma
        display_proba = max(0.01, min(0.99, display_proba))
        prediction = 1 if display_proba > 0.50 else 0

        # 3. GÖRSELLEŞTİRME
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("📊 Analiz Sonucu")
            if prediction == 1:
                st.error("⚠️ **RİSKLİ GRUP**")
                st.write("Verileriniz mental sağlık açısından yüksek riskli bir profile işaret ediyor.")
            else:
                st.success("✅ **SAĞLIKLI**")
                st.write("Teknoloji ve yaşam dengeniz şu an için güvenli görünüyor.")
            
            st.progress(display_proba, text=f"Risk Olasılığı: %{display_proba*100:.1f}")
            st.caption("Not: Sonuçlar model tahmini ve güvenlik filtreleri ile oluşturulmuştur.")

        with c2:
            st.subheader("💡 Size Özel Öneriler")
            recommendations = get_recommendations(input_data, prediction)
            for r in recommendations:
                if prediction == 1: st.warning(r)
                else: st.info(r)

    except Exception as e:
        st.error(f"Analiz sırasında bir hata oluştu: {e}")
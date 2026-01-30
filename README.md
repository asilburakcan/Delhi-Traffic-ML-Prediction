## Delhi Traffic Density Prediction

This is a simple machine learning project designed to predict traffic density in Delhi. If you are interested in traffic analysis, data science, or machine learning, you can find good examples here.

## About the Dataset

The dataset used in this project was obtained from **Kaggle**. It contains real data about trips between different areas in Delhi city.

**Dataset link**: You can search on Kaggle with keywords like "Delhi Traffic" or similar terms.

### Columns in the Dataset:
- **Trip_ID**: Unique identifier for each trip
- **start_area**: Starting point (e.g., Vasant Kunj, Rohini)
- **end_area**: Destination point (e.g., Kalkaji, Dwarka)
- **distance_km**: Trip distance (kilometers)
- **time_of_day**: Time of day (Morning Peak, Afternoon, Evening Peak, Night)
- **day_of_week**: Weekday or weekend (Weekday, Weekend)
- **weather_condition**: Weather condition (Clear, Rain, Fog, Heatwave)
- **traffic_density_level**: Traffic density level - **TARGET VARIABLE** (Low, Medium, High, Very High)
- **road_type**: Road type (Main Road, Highway, Inner Road)
- **average_speed_kmph**: Average speed (km/hour)

## Project Purpose

To put it simply: With the data we have (distance, weather, time, road type, etc.), we are trying to predict **how dense the traffic will be**.

For example: When we say "a 15 km road, in rainy weather, during morning peak hours, on a main road," the model tells us "In this situation, traffic will probably be HIGH (dense)."

## Files

There are 3 main Python files in the project:

### 1. `wwdelhi_scaler.py` - Main Model File

**What does it do?**
- Loads the dataset (`wwdelhi_traffic.csv`)
- Cleans and prepares the data
- Trains a **Logistic Regression** model
- Measures model performance (accuracy, confusion matrix)
- Makes predictions for a new trip

**Why Logistic Regression?**
As written in the comments at the end of the file: "We used logistic regression here because we are trying to answer the question 'is there traffic or not?'" Actually, this is a multi-class classification (4 categories: Low, Medium, High, Very High), but the logic is simple: we categorize.

**Important techniques:**
- **Dummy Variables**: Converts categorical variables (weather, time of day) to numerical form
- **StandardScaler**: Normalizes numerical values (distance, speed) - this is very important!
- **Train-Test Split**: 80% of data for training, 20% for testing
- **Cross-Validation**: Overfitting control with 5-fold cross-validation

**Outputs:**
```
Train Accuracy: 90%
Test Accuracy: 88%
5-Fold CV Accuracy: 87%
```

If the difference between train and test accuracy is more than 10%, it gives a warning (meaning there is overfitting).

**What happens in the code step by step:**
1. Read the CSV file
2. Check for missing values (there are none, but checking is essential!)
3. Group rare weather conditions as 'Other' (overfitting prevention)
4. Convert categorical columns to dummy variables
5. Convert target variable (traffic_density_level) to numerical: Low=0, Medium=1, High=2, Very High=3
6. Split data into train-test sets
7. Scale numerical columns (distance, speed)
8. Train the model
9. Measure performance
10. Make a prediction on a new example and show probabilities

### 2. `ww-p-value-delhi.py` - Statistical Analysis

**What does it do?**
- Uses **Ordered Logit Model** (statsmodels library)
- Shows p-values (which variables are statistically significant? Greater or less than 0.05?)
- Provides summary output (coefficients, standard errors, z-scores)

**When is it used?**
- If you want to scientifically see which factors really affect traffic
- If you need statistical results for an academic report/thesis
- If you want to interpret model coefficients

**What is the difference?**
- `wwdelhi_scaler.py` → For practical prediction (accuracy matters)
- `ww-p-value-delhi.py` → For statistical significance (which variable is important?)

### 3. `testing.py` - Test the Model

**What does it do?**
- Imports the trained model (from `wwdelhi_scaler` module)
- Creates a new sample data
- Makes a prediction

**Example usage:**
```python
new_trip = pd.DataFrame({
    'distance_km': [15.5],              # 15.5 km road
    'average_speed_kmph': [25.0],       # Average 25 km/h speed
    'weather_condition': ['Rain'],      # Rainy weather
    'time_of_day': ['Morning Peak'],    # Morning peak time
    'day_of_week': ['Weekday'],         # Weekday
    'road_type': ['Main Road']          # Main road
})
```

**Output:**
```
Prediction: High

Probabilities:
  Low: 5.23%
  Medium: 18.67%
  High: 61.45%
  Very High: 14.65%
```

The model says: "Under these conditions, traffic will be HIGH (dense) with a 61% probability!"

## How to Use?

### Requirements

```bash
pip install pandas numpy scikit-learn statsmodels
```

### Step 1: Download the dataset
Download the Delhi traffic dataset from Kaggle and save it as `wwdelhi_traffic.csv`.

### Step 2: Run the main model
```bash
python wwdelhi_scaler.py
```

This command:
- Trains the model
- Shows performance metrics
- Makes a sample prediction

### Step 3: Make your own prediction
```bash
python testing.py
```

Or change the values in `testing.py` to test your own scenarios!

### Step 4 (Optional): Statistical analysis
```bash
python ww-p-value-delhi.py
```

To see p-values and coefficients.

## Technical Details

### Data Preprocessing

**1. Categorical Variables → Dummy Variables**
```python
# Example: weather_condition
# ['Clear', 'Rain', 'Fog'] 
# Converts to →
# weather_condition_Rain: 0 or 1
# weather_condition_Fog: 0 or 1
# (Clear: both are 0 - reference category)
```

**2. Numerical Variables → Scaling**
```python
# Distance: ranges from 2.15 km to 26.64 km
# Speed: ranges from 7.6 km/h to 68.5 km/h
# Standardize them: mean=0, std=1
```

**Why?** Logistic Regression is sensitive to distances. The difference between 20 km and 40 km needs to be normalized for the model to work correctly.

### Model Evaluation

**Confusion Matrix** - Shows how well the model works:
```
              Prediction
Actual    Low  Med  High  VHigh
Low       [120  10   2     0  ]
Medium    [ 15  95  18     1  ]
High      [  3  22  88     5  ]
Very High [  1   2  12    45  ]
```

Each row: actual value
Each column: model's prediction

**Classification Report** - Detailed metrics:
- **Precision**: Of the places the model says "High", how many are actually High?
- **Recall**: Of the actual Highs, how many did it catch?
- **F1-Score**: Harmonic mean of Precision and Recall

### Overfitting Control

```python
if train_acc - test_acc > 0.10:
    print("WARNING: Overfitting detected.")
```

If it is very successful on training data but poor on test data → it means the model memorized rather than generalized!

**Solutions:**
- Collect more data
- Reduce the number of features
- Add regularization (L1, L2)
- Use cross-validation (we already do!)

## Tips and Notes

1. **Rare categories**
   ```python
   rare_weather = df['weather_condition'].value_counts()[
       df['weather_condition'].value_counts() < 5
   ].index
   ```
   We make weather conditions with fewer than 5 examples 'Other'. Why? Because the model cannot learn with 3-4 examples, it only memorizes.

2. **Pipeline usage**
   ```python
   model = Pipeline([
       ('preprocessor', preprocessor),
       ('classifier', LogisticRegression())
   ])
   ```
   This is a very clean approach! First preprocessing, then model - everything in one object.

3. **Be careful when making predictions for new data**
   ```python
   # Add columns that were not in training
   for col in X_train.columns:
       if col not in new_trip.columns:
           new_trip[col] = 0
   
   # Arrange columns in the same order. I had a lot of problems here
   new_trip = new_trip[X_train.columns]
   ```
   This is VERY important! Otherwise, the model will give an error.

## Learning Resources

If you want to learn more about these topics:

- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **Logistic Regression**: Classification algorithm
- **Feature Engineering**: Variable engineering
- **Cross-Validation**: Model validation techniques
- **Confusion Matrix**: Model evaluation

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.




TÜRKÇE


# Delhi Trafik Yoğunluğu Tahmini 

Merhaba! Bu proje Delhi'deki trafik yoğunluğunu tahmin etmek için hazırlanmış basit bir makine öğrenmesi projesidir. Eğer trafik analizi, veri bilimi veya makine öğrenmesiyle ilgileniyorsan, buradan güzel örnekler bulabilirsin!

## Veri Seti Hakkında

Bu projede kullanılan veri seti **Kaggle**'dan alınmıştır. Delhi şehrindeki farklı bölgeler arasındaki yolculukları içeren gerçek verilerdir.

**Veri seti linki**: Kaggle'da "Delhi Traffic" veya benzeri anahtar kelimelerle aratabilirsiniz.

### Veri Setindeki Kolonlar:
- **Trip_ID**: Her yolculuğun benzersiz kimliği
- **start_area**: Başlangıç noktası (örn: Vasant Kunj, Rohini)
- **end_area**: Varış noktası (örn: Kalkaji, Dwarka)
- **distance_km**: Yolculuk mesafesi (kilometre)
- **time_of_day**: Günün hangi saati (Morning Peak, Afternoon, Evening Peak, Night)
- **day_of_week**: Hafta içi mi hafta sonu mu (Weekday, Weekend)
- **weather_condition**: Hava durumu (Clear, Rain, Fog, Heatwave)
- **traffic_density_level**: Trafik yoğunluğu seviyesi - **TAHMİN EDİLECEK DEĞER** (Low, Medium, High, Very High)
- **road_type**: Yol tipi (Main Road, Highway, Inner Road)
- **average_speed_kmph**: Ortalama hız (km/saat)

## Proje Amacı

Basitçe söylemek gerekirse: Elimizdeki verilerle (mesafe, hava durumu, saat, yol tipi vb.) **trafiğin ne kadar yoğun olacağını** tahmin etmeye çalışıyoruz.

Örneğin: "15 km'lik bir yol, yağmurlu havada, sabah yoğun saatlerinde, ana yoldan gidilecek" dediğimizde model bize "Bu durumda trafik muhtemelen HIGH (yoğun) olacak" diyor.

## 📂 Dosyalar

Projede 3 ana Python dosyası var:

### 1. `wwdelhi_scaler.py` - Ana Model Dosyası

**Ne yapıyor?**
- Veri setini yüklüyor (`wwdelhi_traffic.csv`)
- Verileri temizliyor ve hazırlıyor
- **Logistic Regression** modeli eğitiyor
- Model performansını ölçüyor (accuracy, confusion matrix)
- Yeni bir yolculuk için tahmin yapıyor

**Neden Logistic Regression?**
Dosyanın sonundaki yorumlarda yazıyor: "Burada trafik var mı yok mu sorusuna cevap aradığımız için logistic regression kullandık." Aslında bu çok sınıflı bir sınıflandırma (4 kategori: Low, Medium, High, Very High), ama mantık basit: kategorilere ayırıyoruz.

**Önemli teknikler:**
- **Dummy Variables**: Kategorik değişkenleri (hava durumu, günün saati) sayısal hale getiriyor
- **StandardScaler**: Sayısal değerleri (mesafe, hız) normalize ediyor - bu çok önemli!
- **Train-Test Split**: Verinin %80'i eğitim, %20'si test için
- **Cross-Validation**: 5 katlı crossvalidation ile overfitting kontrolü

**Çıktılar:**
```
Train Accuracy: ~%90
Test Accuracy: ~%88
5-Fold CV Accuracy: ~%87
```

Eğer train ve test accuracy arasındaki fark %10'dan fazlaysa ⚠️ uyarı veriyor (overfitting var demektir).

**Kodda ne oluyor adım adım:**
1. CSV dosyasını oku
2. Eksik değerleri kontrol et (yok ama kontrol şart!)
3. Nadir görülen hava durumlarını 'Other' olarak grupla (overfitting önlemi)
4. Kategorik kolonları dummy değişkenlere çevir
5. Hedef değişkeni (traffic_density_level) sayısal hale getir: Low=0, Medium=1, High=2, Very High=3
6. Veriyi train-test olarak ayır
7. Sayısal kolonları (mesafe, hız) ölçeklendir
8. Modeli eğit
9. Performansı ölç
10. Yeni bir örnek üzerinde tahmin yap ve olasılıkları göster

### 2. `ww-p-value-delhi.py` - İstatistiksel Analiz

**Ne yapıyor?**
- **Ordered Logit Model** kullanıyor (statsmodels kütüphanesi)
- P-değerlerini gösteriyor (hangi değişkenler istatistiksel olarak anlamlı? 0.05'ten büyük mü küçük mü?)
- Summary çıktısı veriyor (katsayılar, standart hatalar, z-skorları)

**Ne zaman kullanılır?**
- Hangi faktörlerin trafiği gerçekten etkilediğini bilimsel olarak görmek istersen
- Akademik bir rapor/tez için istatistiksel sonuçlara ihtiyacın varsa
- Model katsayılarını yorumlamak istersen

**Fark nedir?**
- `wwdelhi_scaler.py` → Pratik tahmin için (accuracy önemli)
- `ww-p-value-delhi.py` → İstatistiksel anlam için (hangi değişken önemli?)

### 3. `testing.py` - Modeli Test Et

**Ne yapıyor?**
- Eğitilmiş modeli import ediyor (`wwdelhi_scaler` modülünden)
- Yeni bir örnek veri oluşturuyor
- Tahmin yapıyor

**Örnek kullanım:**
```python
new_trip = pd.DataFrame({
    'distance_km': [15.5],              # 15.5 km yol
    'average_speed_kmph': [25.0],       # Ortalama 25 km/h hız
    'weather_condition': ['Rain'],      # Yağmurlu hava
    'time_of_day': ['Morning Peak'],    # Sabah yoğun saati
    'day_of_week': ['Weekday'],         # Hafta içi
    'road_type': ['Main Road']          # Ana yol
})
```

**Çıktı:**
```
Tahmin: High

İhtimaller:
  Low: 5.23%
  Medium: 18.67%
  High: 61.45%
  Very High: 14.65%
```

Model şöyle diyor: "Bu koşullarda trafik %61 ihtimalle HIGH (yoğun) olacak!"

## Nasıl Kullanılır?

### Gereksinimler

```bash
pip install pandas numpy scikit-learn statsmodels
```

### Adım 1: Veri setini indir
Kaggle'dan Delhi trafik veri setini indir ve `wwdelhi_traffic.csv` olarak kaydet.

### Adım 2: Ana modeli çalıştır
```bash
python wwdelhi_scaler.py
```

Bu komut:
- Modeli eğitir
- Performans metriklerini gösterir
- Örnek bir tahmin yapar

### Adım 3: Kendi tahminini yap
```bash
python testing.py
```

Veya `testing.py` içindeki değerleri değiştirerek kendi senaryolarını test et!

### Adım 4 (Opsiyonel): İstatistiksel analiz
```bash
python ww-p-value-delhi.py
```

P-değerlerini ve katsayıları görmek için.

## Teknik Detaylar

### Veri Ön İşleme

**1. Kategorik Değişkenler → Dummy Variables**
```python
# Örnek: weather_condition
# ['Clear', 'Rain', 'Fog'] 
# Dönüşür →
# weather_condition_Rain: 0 veya 1
# weather_condition_Fog: 0 veya 1
# (Clear: her ikisi de 0 olur - referans kategori)
```

**2. Sayısal Değişkenler → Scaling**
```python
# Mesafe: 2.15 km - 26.64 km arası
# Hız: 7.6 km/h - 68.5 km/h arası
# Bunları standartlaştır: ortalama=0, std=1
```

**Neden?** Logistic Regression mesafelere karşı hassastır. 20 km ile 40 km arasındaki fark, modelin doğru çalışması için normalize edilmeli.

### Model Değerlendirme

**Confusion Matrix** - Modelin ne kadar iyi çalıştığını gösterir:
```
              Tahmin
Gerçek    Low  Med  High  VHigh
Low       [120  10   2     0  ]
Medium    [ 15  95  18     1  ]
High      [  3  22  88     5  ]
Very High [  1   2  12    45  ]
```

Her satır: gerçek değer
Her sütun: modelin tahmini

**Classification Report** - Detaylı metrikler:
- **Precision**: Modelin "High" dediği yerlerin kaçı gerçekten High?
- **Recall**: Gerçek High olanların kaçını yakaladı?
- **F1-Score**: Precision ve Recall'un harmonik ortalaması

### Overfitting Kontrolü

```python
if train_acc - test_acc > 0.10:
    print("WARNING: Overfitting detected.")
```

Eğer eğitim verisi üzerinde çok başarılı ama test verisinde kötüyse → model ezberlemişx generalleştirmemiş demektir!

**Çözümler:**
- Daha fazla veri topla
- Feature sayısını azalt
- Regularization ekle (L1, L2)
- Cross-validation kullan (zaten yapıyoruz!)

## İpuçları ve Notlar

1. **Rare kategoriler**
   ```python
   rare_weather = df['weather_condition'].value_counts()[
       df['weather_condition'].value_counts() < 5
   ].index
   ```
   5'ten az örneği olan hava durumlarını 'Other' yapıyoruz. Neden? Çünkü model 3-4 örnekle öğrenemez, sadece ezberler.

2. **Pipeline kullanımı**
   ```python
   model = Pipeline([
       ('preprocessor', preprocessor),
       ('classifier', LogisticRegression())
   ])
   ```
   Bu çok temiz bir yaklaşım! Önce preprocessing, sonra model - her şey tek bir nesnede.

1. **Yeni veri için tahmin yaparken dikkat edin**
   ```python
   # Eğitimde olmayan kolonları ekle
   for col in X_train.columns:
       if col not in new_trip.columns:
           new_trip[col] = 0
   
   # Aynı sırada kolonları düzenle. burada çok sorun yaşamıştım
   new_trip = new_trip[X_train.columns]
   ```
   Bu ÇOOK önemli! Yoksa model hata verir.

## Öğrenme Kaynakları

Eğer bu konularda daha fazla öğrenmek istersen:

- **Pandas**: Veri manipülasyonu
- **Scikit-learn**: Makine öğrenmesi
- **Logistic Regression**: Sınıflandırma algoritması
- **Feature Engineering**: Değişken mühendisliği
- **Cross-Validation**: Model doğrulama teknikleri
- **Confusion Matrix**: Model değerlendirme


## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.



**Not**: Bu bir eğitim/öğrenme projesidir. Gerçek trafik tahminleri için daha karmaşık modeller ve daha fazla veri gerekebilir. 

Sorularınız veya önerileriniz varsa issue açmaktan çekinmeyin! 

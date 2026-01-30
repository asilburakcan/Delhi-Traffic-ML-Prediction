import wwdelhi_scaler as ww
import pandas as pd

# burada örnek bir yolculuk set edelim
new_trip = pd.DataFrame({
    'distance_km': [15.5],
    'average_speed_kmph': [25.0],
    'weather_condition': ['Rain'],
    'time_of_day': ['Morning Peak'],
    'day_of_week': ['Weekday'],
    'road_type': ['Main Road']
})

# yeni yolculuğun dummyleri
new_trip = pd.get_dummies(new_trip, columns=['weather_condition', 'time_of_day', 'day_of_week', 'road_type'], drop_first=True)

# Eğitimde olmayan kolonları ekle (0 değerleriyle)
for col in ww.X_train.columns:
    if col not in new_trip.columns:
        new_trip[col] = 0

# Aynı sırada kolonları düzenle
new_trip = new_trip[ww.X_train.columns]

# Tahminlemebölümü
prediction = ww.model.predict(new_trip)
prediction_proba = ww.model.predict_proba(new_trip)

label_map = {0:'Low', 1:'Medium', 2:'High', 3:'Very High'}
print(f"\ntahmin: {label_map[prediction[0]]}")
print(f"\İihtimaller:")
for i, prob in enumerate(prediction_proba[0]):
    print(f"  {label_map[i]}: {prob:.2%}")
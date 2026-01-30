import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import StandardScaler

# Veri yükle
df = pd.read_csv("wwdelhi_traffic.csv")
print(df.head())
# X hazırlığı
X_p = df.drop(columns=['Trip_ID', 'traffic_density_level','start_area', 'end_area'])
categorical_cols = ['weather_condition', 'time_of_day', 'day_of_week', 'road_type']

# Dummy encoding bölümü
X_p = pd.get_dummies(X_p, columns=categorical_cols, drop_first=True)

# Scaling numeric kolonlar
scaler = StandardScaler()
X_p[['distance_km','average_speed_kmph']] = scaler.fit_transform(X_p[['distance_km','average_speed_kmph']])

# Tüm kolonları float64 yapıyorum sorun yaşamamak için
X_p = X_p.astype('float64')

# y hazırlığı
y_p = df['traffic_density_level'].map({'Low':0, 'Medium':1, 'High':2, "Very High":3})

# Ordered Logit Model
model = OrderedModel(y_p, X_p, distr='logit')
res = model.fit(method='bfgs')
print(res.summary())

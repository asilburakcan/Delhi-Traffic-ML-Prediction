import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("wwdelhi_traffic.csv", na_values=[""])


"""
dummies:
weather_condition
time_of_day
distance_km (4 level yeni sütun)
day_of_week
road_type

y=
traffic_density_level
"""


#EDA bölümü

print(df.head(10))
print(df.info())
print(df.shape)
print(df["distance_km"].max(), df["distance_km"].min())


# Eksik değer kontrolü
print(df.isnull().sum())

# negatif veya sıfır hız kontrolü
print((df['average_speed_kmph'] <= 0).sum())


# nadir weather combine. FREKANSA BAK. DÜŞÜK İSE OVERFİTTİNG / RANDOM FORREST XRGB GİBİ MODELLERDE DOKUNMA
rare_weather = df['weather_condition'].value_counts()[df['weather_condition'].value_counts() < 5].index
df['weather_condition'] = df['weather_condition'].replace(rare_weather, 'Other')

#Dummy değişkenler oluşturmak
X = df.drop(columns=['Trip_ID', 'traffic_density_level','start_area', 'end_area'])
categorical_cols = ['weather_condition', 'time_of_day', 'day_of_week', 'road_type']
print(df.columns)

X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

#hedef
y = df['traffic_density_level'].map({'Low':0, 'Medium':1, 'High':2, "Very High":3})


print(X)
print(df.info())
print(X.isna().sum())
print(y.isna().sum())  # 0 olmalı
print(X.shape, y.shape) 

# Train-test split bölümü sadece scale olmaz. 
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#scaling - sadece numeric kolonlar
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_features = ['distance_km', 'average_speed_kmph']

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numeric_features)
    ],
    remainder='passthrough'  # Dummy kolonlar dokunulmadan geçer
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Logistic Regression

model.fit(X_train, y_train)

# Train ve Test accuracy
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\nTrain Accuracy: {train_acc:.2%}")
print(f"Test Accuracy:  {test_acc:.2%}")
print(f"Overfitting Gap: {(train_acc - test_acc):.2%}")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Overfitting kontrolü
if train_acc - test_acc > 0.10:
    print("\n!!!!!!!Overfitting detected (gap > 10%)")
else:
    print("\nModel generalizes well")

# Test set üzerinde confusion matrix
print("\nTest Set Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred_test, 
                          target_names=['Low','Medium','High','Very High']))









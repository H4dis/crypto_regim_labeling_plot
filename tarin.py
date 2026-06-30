import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from feature_engineering import create_advanced_features, FEATURES

df = pd.read_csv("data/raw/CLUDE_labeled_with_labels.csv")
df_advanced = create_advanced_features(df)

X = df_advanced[FEATURES]
y = df_advanced['price_sharp_increasing']

mask = X.notna().all(axis=1)
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, 'models/price_sharp_model.pkl')

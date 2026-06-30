import pandas as pd
import joblib
from pathlib import Path
from feature_engineering import create_advanced_features, FEATURES


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

csv_files = list(DATA_DIR.glob("*.csv"))
latest_file = csv_files[0]
print(f"file found {latest_file.name}")


model = joblib.load(MODEL_DIR / 'price_sharp_model.pkl')

df_new = pd.read_csv(latest_file)


df_advanced = create_advanced_features(df_new)

X_new = df_advanced[FEATURES].fillna(0)
predictions = model.predict(X_new)

#prdict data frame
df_new['predicted_sharp_increase'] = predictions


# filter pos dataframe
sharp_increases = df_new[df_new['predicted_sharp_increase'] == 1]
print(f"\n🔍 sharp increase predicted: {len(sharp_increases)}")
if len(sharp_increases) > 0:
    print(sharp_increases[['datetime_utc', 'SS_price_hourly', 'SS_volume_hourly']])

# trading signals
def trading_signal(row):
    if row['predicted_sharp_increase'] == 1:
        return "BUY (Increase Expected)"
    elif row.get('price_sharp_increasing', 0) == 1:
        return " SELL (Increase Happened)"
    else:
        return "HOLD"

df_new['signal'] = df_new.apply(trading_signal, axis=1)
print("\n sinals:")
print(df_new[['datetime_utc', 'signal']].head(10))

# 3final signal
output_path = DATA_DIR / "final_signals.csv"
df_new.to_csv(output_path, index=False)
print(f"\nfinal signal saved: {output_path}")

# plot visual
print("\n prdictions:")
print(f"num of rows: {len(df_new)}")
print(f"predict big rise {predictions.sum()}")
print(f"signal BUY: {(df_new['signal'] == 'BUY (Increase Expected)').sum()}")
print(f"signal SELL: {(df_new['signal'] == 'SELL (Increase Happened)').sum()}")
print(f"signal HOLD: {(df_new['signal'] == 'HOLD').sum()}")






import pandas as pd
import plotly.express as px

df = pd.read_csv("data/raw/final_signals.csv")
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

# prcie and signal plot
fig = px.scatter(df, x='datetime_utc', y='SS_price_hourly',
                 color='signal', title='Price with Trading Signals')
fig.show()
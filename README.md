# Crypto Regime Labeling & Signal Detection

## 📊 دیتاست
- ۲,۱۶۰ رکورد از داده‌های بیت‌کوین
- تایم‌فریم: ساعتی
- بازه: آوریل ۲۰۲۶

## 🤖 مدل
- Random Forest با ۲۰۰ درخت
- دقت: ۹۹٪ روی داده‌های تست
- ویژگی‌های کلیدی: return_1, return_5, slope_5

## 📈 سیگنال‌ها
- ۳۲ سیگنال خرید (BUY)
- ۲,۱۲۸ سیگنال نگهداری (HOLD)

## 🚀 نحوه استفاده
```python
import pandas as pd
df = pd.read_csv('final_signals.csv')
buy_signals = df[df['signal'] == 'BUY']

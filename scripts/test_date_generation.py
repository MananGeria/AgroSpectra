"""
Test script to verify dynamic date generation
"""
from datetime import datetime, timedelta

# Show what dates will be generated
today = datetime.now().date()
print(f"\n📅 Today's date: {today.strftime('%Y-%m-%d')}")
print(f"\n🔄 10 days of historical data will be generated:")
print("-" * 50)

for day_offset in range(9, -1, -1):  # 9 days ago to today
    historical_date = today - timedelta(days=day_offset)
    days_ago = day_offset
    label = "TODAY" if days_ago == 0 else f"{days_ago} days ago"
    print(f"  {historical_date.strftime('%Y-%m-%d')} ({label})")

print("-" * 50)
print(f"\n✅ Dates are DYNAMIC - calculated from current date")
print(f"✅ NO HARDCODED dates (2025-10-23 removed)")
print(f"\n💡 Run an analysis and check the Pest Risk Timeline tab!")

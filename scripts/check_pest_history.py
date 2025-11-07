"""
Script to check pest risk history in database
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tier6_storage.database import DatabaseManager

def check_pest_history():
    """Check pest risk history records"""
    db = DatabaseManager()
    db.connect()
    
    cursor = db.conn.cursor()
    
    # Count total records
    cursor.execute('SELECT COUNT(*) FROM pest_risk_history')
    total_count = cursor.fetchone()[0]
    print(f"\n📊 Total pest risk history records: {total_count}")
    
    if total_count > 0:
        # Show latest 10 records
        cursor.execute("""
            SELECT date, latitude, longitude, pest_risk, 
                   temperature_c, humidity_pct, ndvi, crop_type
            FROM pest_risk_history 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        
        rows = cursor.fetchall()
        
        print("\n📋 Latest 10 records:")
        print("-" * 100)
        print(f"{'Date':<12} {'Lat':<10} {'Lon':<10} {'Pest Risk':<12} {'Temp(°C)':<10} {'Humidity(%)':<12} {'NDVI':<8} {'Crop':<10}")
        print("-" * 100)
        
        for row in rows:
            print(f"{row[0]:<12} {row[1]:<10.4f} {row[2]:<10.4f} {row[3]*100:<11.1f}% {row[4]:<10.1f} {row[5]:<12.1f} {row[6]:<8.3f} {row[7]:<10}")
        
        print("-" * 100)
        
        # Show unique locations
        cursor.execute("""
            SELECT DISTINCT location_hash, COUNT(*) as record_count
            FROM pest_risk_history
            GROUP BY location_hash
        """)
        
        locations = cursor.fetchall()
        print(f"\n📍 Unique locations tracked: {len(locations)}")
        for loc in locations:
            print(f"   - {loc[0]}: {loc[1]} records")
    else:
        print("\n⚠️ No pest risk history records found.")
        print("💡 Run an analysis to start collecting historical data!")
    
    db.close()
    print("\n✅ Database check complete!\n")

if __name__ == "__main__":
    check_pest_history()

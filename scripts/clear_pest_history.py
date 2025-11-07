"""
Clear old pest risk history data to test new dynamic date generation
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tier6_storage.database import DatabaseManager

def clear_pest_history():
    """Clear all pest risk history records"""
    db = DatabaseManager()
    db.connect()
    
    cursor = db.conn.cursor()
    
    # Count before
    cursor.execute('SELECT COUNT(*) FROM pest_risk_history')
    count_before = cursor.fetchone()[0]
    print(f"\n📊 Records before clearing: {count_before}")
    
    # Clear all records
    if count_before > 0:
        response = input("\n⚠️  Delete all pest risk history records? (yes/no): ")
        if response.lower() == 'yes':
            cursor.execute('DELETE FROM pest_risk_history')
            db.conn.commit()
            
            cursor.execute('SELECT COUNT(*) FROM pest_risk_history')
            count_after = cursor.fetchone()[0]
            print(f"✅ Records after clearing: {count_after}")
            print("🔄 Database cleared! Run a new analysis to generate fresh data.")
        else:
            print("❌ Operation cancelled.")
    else:
        print("✅ Database is already empty!")
    
    db.close()

if __name__ == "__main__":
    clear_pest_history()

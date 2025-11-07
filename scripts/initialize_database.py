"""
Database Initialization Script
Run this script to set up the database structure
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from tier6_storage.database import initialize_database
from loguru import logger

# Configure logging
logger.add(
    "logs/initialization.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO"
)


def main():
    """Initialize database"""
    logger.info("Starting database initialization")
    
    try:
        # Initialize database
        initialize_database()
        
        logger.info("✅ Database initialized successfully!")
        print("\n✅ Database initialized successfully!")
        print(f"Database location: data/storage/agrospectra.db")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

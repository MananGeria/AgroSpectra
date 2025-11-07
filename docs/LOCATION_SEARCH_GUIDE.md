# Location Search Feature - User Guide

## 🔍 Search by Location Name (NEW!)

AgroSpectra now supports **3 ways to define your area of interest**:

### 1️⃣ Search by Name (Easiest!) ✨NEW✨

**How to use:**

1. In the sidebar, find "🔍 Search by Location Name"
2. Type any location name in the search box
3. Click "🔍 Search Location"
4. The app will automatically:
   - Find the location on the map
   - Create a 10km x 10km area around it
   - Show you the exact coordinates

**Example searches:**

- `Punjab, India`
- `Iowa, USA`
- `São Paulo, Brazil`
- `New Delhi`
- `California Central Valley`
- `Maharashtra Farmland`
- `Uttar Pradesh`

**Search Tips:**

- Be specific: Include country or state for better results
- Use English names
- If not found, try variations (e.g., "Mumbai" vs "Bombay")

**What happens:**

- ✅ Location found → Automatic 10km x 10km bounding box created
- ❌ Not found → Try a different search term or use manual coordinates

### 2️⃣ Manual Coordinates (Precise!)

**How to use:**

1. Select "Manual Coordinates" radio button
2. Enter bounding box coordinates:
   - **Min Latitude** (bottom edge)
   - **Min Longitude** (left edge)
   - **Max Latitude** (top edge)
   - **Max Longitude** (right edge)
3. Click "✅ Set AOI"
4. The area is now saved and ready for analysis

**Fixed Issue:**
Previously, manual coordinates weren't being saved to session state. Now they are **properly saved** and persist until you change them or restart the app.

**Example (Punjab, India):**

```
Min Latitude:  30.50
Min Longitude: 75.50
Max Latitude:  31.00
Max Longitude: 76.00
```

### 3️⃣ Draw on Map (Visual!)

**How to use:**

1. Click "🗺️ Open Map Selector" button in the main area
2. Use drawing tools to outline your field
3. Close the map selector
4. Area is automatically saved

## 🆚 Comparison

| Method                 | Best For                         | Pros                        | Cons                     |
| ---------------------- | -------------------------------- | --------------------------- | ------------------------ |
| **Location Search**    | Quick analysis, known places     | Fast, no coordinates needed | ~10km area (not precise) |
| **Manual Coordinates** | Precise areas, known coordinates | Exact bounding box control  | Need to know coordinates |
| **Draw on Map**        | Visual field selection           | Intuitive, see the area     | Requires map interaction |

## 🌍 Location Search Technology

**How it works:**

- Uses **Nominatim API** (OpenStreetMap)
- **Free** - No API key required
- **Global coverage** - Works anywhere in the world
- **Automatic** - Just type and search

**Search Scope:**

- Cities, towns, villages
- States, provinces, regions
- Countries
- Landmarks
- Postal codes (in some countries)

**Privacy:**

- No personal data stored
- Search queries not logged
- Uses public OpenStreetMap database

## 🐛 Troubleshooting

### Issue: "Location not found"

**Causes:**

1. Spelling error
2. Location too small or obscure
3. Non-English name without translation

**Solutions:**

- Try broader search (state instead of village)
- Include country name ("Amritsar, India")
- Use English spelling
- Try nearby major city instead

### Issue: "Search failed"

**Causes:**

1. No internet connection
2. Nominatim API temporarily unavailable

**Solutions:**

- Check internet connection
- Wait a few seconds and try again
- Use manual coordinates as fallback

### Issue: Manual coordinates not saving

**Fixed!** This was a bug where coordinates weren't stored in session state. Now they are properly saved when you click "✅ Set AOI".

**To verify it's working:**

1. Enter coordinates
2. Click "✅ Set AOI"
3. You should see "✅ AOI set successfully!"
4. Scroll down - you should see area summary with your coordinates

## 💡 Pro Tips

### For Indian Users

- Search examples:
  - `Punjab India` → Great for wheat belt
  - `Maharashtra Nashik` → Grape vineyards
  - `Tamil Nadu Thanjavur` → Rice paddies
  - `Gujarat Saurashtra` → Cotton region

### For US Users

- Search examples:
  - `Iowa Corn Belt`
  - `California Central Valley`
  - `Kansas Wheat Fields`
  - `Nebraska Agriculture`

### For Other Regions

- Search examples:
  - `São Paulo Brazil` → Sugarcane
  - `Ukraine Wheat Region`
  - `Australia Wheat Belt`
  - `Argentina Pampas`

## 🚀 Quick Start Examples

### Example 1: Analyze Punjab Wheat Fields

```
1. Search: "Punjab, India"
2. Click "🔍 Search Location"
3. Select crop: "Wheat"
4. Click "🔍 Run Analysis"
5. See ICAR enhancements for India! 🇮🇳
```

### Example 2: Monitor Iowa Corn

```
1. Search: "Iowa, USA"
2. Click "🔍 Search Location"
3. Select crop: "Maize"
4. Click "🔍 Run Analysis"
5. Get global satellite + weather analysis
```

### Example 3: Precise Field Monitoring

```
1. Select "Manual Coordinates"
2. Enter your exact field boundaries:
   Min Lat: 30.7340
   Min Lon: 75.8400
   Max Lat: 30.7440
   Max Lon: 75.8600
3. Click "✅ Set AOI"
4. Select crop and run analysis
```

## 📊 Search Statistics

**Coverage:**

- 🌍 Global: 190+ countries
- 🏙️ Cities: 1+ million locations
- 🗺️ Accuracy: ±1-5 km (depends on location)

**Search Speed:**

- Average: 1-3 seconds
- Depends on internet speed
- Cached results load instantly

## 🔐 Privacy & Data

**What we collect:**

- ❌ NO search history stored
- ❌ NO location tracking
- ❌ NO personal data saved

**What we use:**

- ✅ Coordinates for analysis only
- ✅ Public OpenStreetMap data
- ✅ Temporary session storage

All location data is deleted when you close the app.

---

**Need Help?**

- Issue Tracker: https://github.com/MananGeria/AgroSpectra/issues
- Documentation: See README.md
- ICAR Guide: See docs/ICAR_SETUP_GUIDE.md

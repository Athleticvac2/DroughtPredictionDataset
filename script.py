import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Kenya's main agricultural regions and their characteristics
regions = {
	'Central Highlands': {
		'counties': ['Nyeri', 'Kirinyaga', 'Murang\'a', 'Kiambu'],
		'soil_types': ['Nitisols', 'Andosols'],
		'crops': ['Tea', 'Coffee', 'Maize', 'Beans', 'Potatoes'],
		'altitude': '1500-2500m',
		'avg_rainfall': '1200-2000mm',
		'temp_range': '10-24'
	},
	'Western Kenya': {
		'counties': ['Kakamega', 'Bungoma', 'Busia', 'Vihiga'],
		'soil_types': ['Ferralsols', 'Acrisols'],
		'crops': ['Maize', 'Sugarcane', 'Beans', 'Sweet Potatoes'],
		'altitude': '1000-1800m',
		'avg_rainfall': '1100-2000mm',
		'temp_range': '14-32'
	},
	'Rift Valley': {
		'counties': ['Uasin Gishu', 'Trans Nzoia', 'Nakuru', 'Nandi'],
		'soil_types': ['Vertisols', 'Planosols'],
		'crops': ['Wheat', 'Maize', 'Pyrethrum', 'Potatoes'],
		'altitude': '1500-2300m',
		'avg_rainfall': '900-1800mm',
		'temp_range': '12-28'
	},
	'Coast': {
		'counties': ['Kilifi', 'Kwale', 'Lamu', 'Mombasa'],
		'soil_types': ['Luvisols', 'Arenosols'],
		'crops': ['Coconuts', 'Cashewnuts', 'Cassava', 'Mangoes'],
		'altitude': '0-500m',
		'avg_rainfall': '500-1200mm',
		'temp_range': '22-35'
	},
	'Eastern': {
		'counties': ['Machakos', 'Makueni', 'Kitui', 'Embu'],
		'soil_types': ['Cambisols', 'Lixisols'],
		'crops': ['Sorghum', 'Millet', 'Green Grams', 'Cowpeas'],
		'altitude': '500-1200m',
		'avg_rainfall': '500-1000mm',
		'temp_range': '18-33'
	}
}

# Crop seasons in Kenya
crop_seasons = {
	'Long Rains': {
		'start_month': 3,  # March
		'end_month': 5,  # May
		'rainfall_boost': 1.5
	},
	'Short Rains': {
		'start_month': 10,  # October
		'end_month': 12,  # December
		'rainfall_boost': 1.2
	}
}

# Soil characteristics
soil_characteristics = {
	'Nitisols': {'water_retention': 0.8, 'fertility': 0.9, 'drainage': 0.85},
	'Andosols': {'water_retention': 0.75, 'fertility': 0.85, 'drainage': 0.8},
	'Ferralsols': {'water_retention': 0.6, 'fertility': 0.7, 'drainage': 0.75},
	'Acrisols': {'water_retention': 0.55, 'fertility': 0.65, 'drainage': 0.7},
	'Vertisols': {'water_retention': 0.9, 'fertility': 0.8, 'drainage': 0.6},
	'Planosols': {'water_retention': 0.85, 'fertility': 0.75, 'drainage': 0.65},
	'Luvisols': {'water_retention': 0.7, 'fertility': 0.75, 'drainage': 0.8},
	'Arenosols': {'water_retention': 0.4, 'fertility': 0.5, 'drainage': 0.9},
	'Cambisols': {'water_retention': 0.65, 'fertility': 0.7, 'drainage': 0.75},
	'Lixisols': {'water_retention': 0.6, 'fertility': 0.65, 'drainage': 0.7}
}


def generate_weather_data(region, date):
	"""Generate realistic weather data based on region and season"""
	month = date.month
	is_rainy_season = False
	rainfall_multiplier = 1.0

	# Check if it's rainy season
	if month in range(crop_seasons['Long Rains']['start_month'],
	                  crop_seasons['Long Rains']['end_month'] + 1):
		is_rainy_season = True
		rainfall_multiplier = crop_seasons['Long Rains']['rainfall_boost']
	elif month in range(crop_seasons['Short Rains']['start_month'],
	                    crop_seasons['Short Rains']['end_month'] + 1):
		is_rainy_season = True
		rainfall_multiplier = crop_seasons['Short Rains']['rainfall_boost']

	# Parse rainfall range
	min_rain, max_rain = map(int, region['avg_rainfall'].replace('mm', '').split('-'))
	min_temp, max_temp = map(int, region['temp_range'].split('-'))

	# Generate weather data
	if is_rainy_season:
		rainfall = np.random.uniform(min_rain / 12 * rainfall_multiplier,
		                             max_rain / 12 * rainfall_multiplier)
		humidity = np.random.uniform(70, 85)
	else:
		rainfall = np.random.uniform(0, min_rain / 12)
		humidity = np.random.uniform(50, 65)

	temperature = np.random.uniform(min_temp, max_temp)

	return rainfall, temperature, humidity


# Generate data
start_date = datetime(2019, 1, 1)
end_date = datetime(2024, 1, 1)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

data = []
for region_name, region_data in regions.items():
	for county in region_data['counties']:
		for soil_type in region_data['soil_types']:
			for crop in region_data['crops']:
				for date in dates:
					rainfall, temp, humidity = generate_weather_data(region_data, date)

					# Get soil characteristics
					soil_chars = soil_characteristics[soil_type]

					# Calculate soil moisture based on rainfall and soil characteristics
					soil_moisture = min(100, (rainfall * soil_chars['water_retention'] +
					                          humidity * 0.3) * (1 - soil_chars['drainage'] * 0.2))

					# Calculate crop health index (0-100)
					crop_health = min(100, max(0,
					                           soil_moisture * 0.4 +
					                           soil_chars['fertility'] * 30 +
					                           (1 - abs(temp - 25) / 25) * 30  # Optimal temp around 25°C
					                           ))

					data.append({
						'Date': date.strftime('%Y-%m-%d'),
						'Region': region_name,
						'County': county,
						'Soil_Type': soil_type,
						'Crop': crop,
						'Rainfall_mm': round(rainfall, 1),
						'Temperature_Celsius': round(temp, 1),
						'Humidity_Percent': round(humidity, 1),
						'Soil_Moisture_Percent': round(soil_moisture, 1),
						'Water_Retention': soil_chars['water_retention'],
						'Soil_Fertility': soil_chars['fertility'],
						'Soil_Drainage': soil_chars['drainage'],
						'Crop_Health_Index': round(crop_health, 1),
						'Altitude_Range': region_data['altitude']
					})

# Convert to DataFrame and export
df = pd.DataFrame(data)
df.to_csv('/home/athleticvac2/PycharmProjects/DroughtPredictionDataset/data/kenya_agricultural_data.csv', index=False)

print(f"Generated {len(df)} records")
print("\nSample of the data:")
print(df.head())
print("\nDataset statistics:")
print(df.describe())
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create a large sample Chicago crime dataset
np.random.seed(42)

# Generate 1 million records to demonstrate performance differences
n_records = 1_000_000

print(f"Generating {n_records:,} records...")

# Generate random dates between 2001 and 2025
start_date = datetime(2001, 1, 1)
end_date = datetime(2025, 1, 1)
date_range = (end_date - start_date).days

dates = [start_date + timedelta(days=int(np.random.random() * date_range))
         for _ in range(n_records)]

# Crime types from Chicago crime dataset
crime_types = [
    'THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'NARCOTICS', 'ASSAULT',
    'BURGLARY', 'MOTOR VEHICLE THEFT', 'ROBBERY', 'DECEPTIVE PRACTICE',
    'CRIMINAL TRESPASS', 'WEAPONS VIOLATION', 'OFFENSE INVOLVING CHILDREN'
]

# Chicago districts (1-25)
districts = list(range(1, 26))

# Generate random data
print("Creating DataFrame...")
data = {
    'ID': range(1, n_records + 1),
    'Case Number': [f'JC{np.random.randint(100000, 999999)}' for _ in range(n_records)],
    'Date': dates,
    'Primary Type': np.random.choice(crime_types, n_records),
    'Description': [f'Description {i}' for i in range(n_records)],
    'Location Description': np.random.choice(['STREET', 'RESIDENCE', 'APARTMENT', 'SIDEWALK',
                                               'OTHER', 'PARKING LOT', 'RESTAURANT', 'ALLEY',
                                               'SCHOOL', 'COMMERCIAL'], n_records),
    'Arrest': np.random.choice([True, False], n_records, p=[0.2, 0.8]),
    'Domestic': np.random.choice([True, False], n_records, p=[0.15, 0.85]),
    'District': np.random.choice(districts, n_records),
    'Ward': np.random.randint(1, 51, n_records),
    'Latitude': np.random.uniform(41.6, 42.0, n_records),
    'Longitude': np.random.uniform(-87.9, -87.5, n_records),
    'Year': [d.year for d in dates],
}

# Create pandas DataFrame
df = pd.DataFrame(data)

# Save as parquet
output_file = 'data/chicago_crime_2001_2025.parquet'
print(f"Writing to {output_file}...")
try:
    df.to_parquet(output_file, index=False, engine='pyarrow')
except ImportError:
    print("PyArrow not available, trying fastparquet...")
    df.to_parquet(output_file, index=False, engine='fastparquet')

file_size = os.path.getsize(output_file) / 1024 / 1024
print(f"\nCreated {output_file}")
print(f"Records: {n_records:,}")
print(f"File size: {file_size:.2f} MB")
print(f"\nFirst few rows:")
print(df.head())

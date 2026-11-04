import pandas as pd
from pathlib import Path
from pybaseball import statcast

# Find the main project folder
project_folder = Path(__file__).resolve().parents[1]

# Create the folder for our raw data
data_folder = project_folder / "data" / "raw"
data_folder.mkdir(parents=True, exist_ok=True)

# Download MLB Statcast data
print("Downloading MLB Statcast data...")

df = statcast(
    start_dt="2024-04-01",
    end_dt="2024-04-03"
)

# Save the original dataset
output_file = data_folder / "statcast_data.csv"
df.to_csv(output_file, index=False)

# Display basic dataset information
print("\nDownload complete!")
print("\nDataset dimensions:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))


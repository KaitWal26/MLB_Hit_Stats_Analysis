
import pandas as pd
from pathlib import Path

# Locate project folders
project_folder = Path(__file__).resolve().parents[1]
raw_file = project_folder / "data" / "raw" / "statcast_data.csv"

# Load the original dataset
df = pd.read_csv(raw_file)

print("Original dataset:", df.shape)

# Keep only pitches resulting in a ball put into play
# and with a recorded exit velocity
batted_balls = df.loc[
    (df["type"] == "X") &
    (df["launch_speed"].notna())
].copy()

# Create our prediction target
# 1 = Hard hit (95 mph or higher)
# 0 = Not hard hit (below 95 mph)
batted_balls["hard_hit"] = (
    batted_balls["launch_speed"] >= 95
).astype(int)

# Display results
print("\nBatted-ball dataset:", batted_balls.shape)

print("\nHard-hit counts:")
print(batted_balls["hard_hit"].value_counts())

print("\nHard-hit percentages:")
print(
    (batted_balls["hard_hit"].value_counts(normalize=True) * 100)
    .round(2)
)

# Examine missing values in our potential predictors
predictors = [
    "release_speed",
    "pitch_type",
    "release_spin_rate",
    "p_throws",
    "stand",
    "balls",
    "strikes",
    "plate_x",
    "plate_z",
    "inning",
    "outs_when_up"
]

print("\nMissing predictor values:")
print(
    batted_balls[predictors]
    .isnull()
    .sum()
    .sort_values(ascending=False)
)

# Save the filtered dataset
processed_folder = project_folder / "data" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)

output_file = processed_folder / "batted_ball_data.csv"
batted_balls.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully!")

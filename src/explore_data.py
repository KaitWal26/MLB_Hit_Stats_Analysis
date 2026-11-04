
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Locate project folders
project_folder = Path(__file__).resolve().parents[1]
data_file = project_folder / "data" / "processed" / "batted_ball_data.csv"

# Create visualization folder
visualization_folder = project_folder / "visualizations"
visualization_folder.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(data_file)

# Set visualization style
sns.set_theme(style="whitegrid")

print("Exploratory Data Analysis")
print("-------------------------")

# 1. Overall hard-hit distribution
print("\nHard-hit distribution:")
print(df["hard_hit"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="hard_hit")
plt.title("Hard-Hit vs. Non-Hard-Hit Batted Balls")
plt.xlabel("Hard Hit (0 = No, 1 = Yes)")
plt.ylabel("Number of Batted Balls")
plt.tight_layout()
plt.savefig(visualization_folder / "hard_hit_distribution.png")
plt.close()

# 2. Hard-hit rate by pitch type
pitch_rates = (
    df.groupby("pitch_type")
    .agg(
        hard_hit_rate=("hard_hit", "mean"),
        sample_size=("hard_hit", "size")
    )
)

# Only include pitch types with at least 30 observations
pitch_rates = pitch_rates[pitch_rates["sample_size"] >= 30]

pitch_rates = pitch_rates.sort_values(
    "hard_hit_rate",
    ascending=False
)

print("\nHard-hit rate by pitch type:")
print(pitch_rates)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=pitch_rates.reset_index(),
    x="pitch_type",
    y="hard_hit_rate"
)
plt.title("Hard-Hit Rate by Pitch Type")
plt.xlabel("Pitch Type")
plt.ylabel("Hard-Hit Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(visualization_folder / "hard_hit_by_pitch_type.png")
plt.close()

# 3. Pitch velocity compared to hard-hit outcomes
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="hard_hit",
    y="release_speed"
)
plt.title("Pitch Velocity by Hard-Hit Outcome")
plt.xlabel("Hard Hit (0 = No, 1 = Yes)")
plt.ylabel("Pitch Velocity (mph)")
plt.tight_layout()
plt.savefig(visualization_folder / "pitch_velocity_by_hard_hit.png")
plt.close()

# 4. Pitch spin rate compared to hard-hit outcomes
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="hard_hit",
    y="release_spin_rate"
)
plt.title("Pitch Spin Rate by Hard-Hit Outcome")
plt.xlabel("Hard Hit (0 = No, 1 = Yes)")
plt.ylabel("Release Spin Rate (rpm)")
plt.tight_layout()
plt.savefig(visualization_folder / "spin_rate_by_hard_hit.png")
plt.close()

print("\nVisualizations saved successfully!")
print("Check your visualizations folder.")

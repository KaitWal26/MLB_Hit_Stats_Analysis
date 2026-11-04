
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency, mannwhitneyu
import numpy as np

# Locate and load the cleaned dataset
project_folder = Path(__file__).resolve().parents[1]
data_file = project_folder / "data" / "processed" / "batted_ball_data.csv"

df = pd.read_csv(data_file)

print("STATISTICAL HYPOTHESIS TESTING")
print("=" * 40)

# Significance threshold
alpha = 0.05


# TEST 1: Pitch type and hard-hit outcomes
print("\nTEST 1: Pitch Type vs. Hard-Hit Outcome")

# Exclude pitch types with fewer than 30 observations
pitch_counts = df["pitch_type"].value_counts()
common_pitches = pitch_counts[pitch_counts >= 30].index

pitch_data = df[df["pitch_type"].isin(common_pitches)]

# Create contingency table
contingency_table = pd.crosstab(
    pitch_data["pitch_type"],
    pitch_data["hard_hit"]
)

chi2, p_value, degrees_freedom, expected = chi2_contingency(
    contingency_table
)

# Calculate Cramer's V (effect size)
n = contingency_table.to_numpy().sum()
minimum_dimension = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n * minimum_dimension))

print("Chi-square statistic:", round(chi2, 4))
print("Degrees of freedom:", degrees_freedom)
print("P-value:", p_value)
print("Cramer's V:", round(cramers_v, 4))

if p_value < alpha:
    print("Result: Statistically significant association.")
else:
    print("Result: No statistically significant association.")


# TEST 2: Pitch velocity
print("\nTEST 2: Pitch Velocity vs. Hard-Hit Outcome")

not_hard_hit_speed = df.loc[
    df["hard_hit"] == 0, "release_speed"
].dropna()

hard_hit_speed = df.loc[
    df["hard_hit"] == 1, "release_speed"
].dropna()

statistic, p_value = mannwhitneyu(
    not_hard_hit_speed,
    hard_hit_speed,
    alternative="two-sided"
)

print("Median velocity (not hard-hit):",
      round(not_hard_hit_speed.median(), 2), "mph")

print("Median velocity (hard-hit):",
      round(hard_hit_speed.median(), 2), "mph")

print("Mann-Whitney U statistic:", round(statistic, 2))
print("P-value:", p_value)

if p_value < alpha:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")


# TEST 3: Pitch spin rate
print("\nTEST 3: Pitch Spin Rate vs. Hard-Hit Outcome")

not_hard_hit_spin = df.loc[
    df["hard_hit"] == 0, "release_spin_rate"
].dropna()

hard_hit_spin = df.loc[
    df["hard_hit"] == 1, "release_spin_rate"
].dropna()

statistic, p_value = mannwhitneyu(
    not_hard_hit_spin,
    hard_hit_spin,
    alternative="two-sided"
)

print("Median spin rate (not hard-hit):",
      round(not_hard_hit_spin.median(), 2), "rpm")

print("Median spin rate (hard-hit):",
      round(hard_hit_spin.median(), 2), "rpm")

print("Mann-Whitney U statistic:", round(statistic, 2))
print("P-value:", p_value)

if p_value < alpha:
    print("Result: Statistically significant difference.")
else:
    print("Result: No statistically significant difference.")

print("\n" + "=" * 40)
print("STATISTICAL ANALYSIS COMPLETE!")

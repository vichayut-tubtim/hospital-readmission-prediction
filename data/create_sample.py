import pandas as pd

df = pd.read_csv(
    "data/diabetic_data.csv"
)

df = df.drop(
    columns=["readmitted"],
    errors="ignore"
)


sample = df.sample(
    10,
    random_state=42
)


sample.to_csv(
    "data/sample_patients.csv",
    index=False
)


print("Created sample_patients.csv")
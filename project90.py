import pandas as pd
import zipfile
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# LOAD DATA
# =========================

zip_path = r"C:\Users\DELL\Downloads\e88186124ec611f1.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    train = pd.read_csv(z.open('dataset/train.csv'))
    test = pd.read_csv(z.open('dataset/test.csv'))

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

# =========================
# HANDLE MISSING VALUES
# =========================

train["RoadType"] = train["RoadType"].fillna(train["RoadType"].mode()[0])
test["RoadType"] = test["RoadType"].fillna(train["RoadType"].mode()[0])

train["Weather"] = train["Weather"].fillna(train["Weather"].mode()[0])
test["Weather"] = test["Weather"].fillna(train["Weather"].mode()[0])

train["Temperature"] = train["Temperature"].fillna(
    train["Temperature"].median()
)

test["Temperature"] = test["Temperature"].fillna(
    train["Temperature"].median()
)

# =========================
# TIME FEATURES
# =========================

train["hour"] = train["timestamp"].str.split(":").str[0].astype(int)
train["minute"] = train["timestamp"].str.split(":").str[1].astype(int)

test["hour"] = test["timestamp"].str.split(":").str[0].astype(int)
test["minute"] = test["timestamp"].str.split(":").str[1].astype(int)

train["time_in_minutes"] = train["hour"] * 60 + train["minute"]
test["time_in_minutes"] = test["hour"] * 60 + test["minute"]

peak_hours = [8, 9, 10, 17, 18, 19]

train["is_peak"] = train["hour"].isin(peak_hours).astype(int)
test["is_peak"] = test["hour"].isin(peak_hours).astype(int)

# =========================
# GEOHASH FEATURES
# =========================

# Frequency encoding
freq = train["geohash"].value_counts()

train["geohash_freq"] = train["geohash"].map(freq)
test["geohash_freq"] = test["geohash"].map(freq).fillna(0)

# Target encoding
geohash_mean = train.groupby("geohash")["demand"].mean()

global_mean = train["demand"].mean()

train["geohash_target"] = train["geohash"].map(geohash_mean)

test["geohash_target"] = test["geohash"].map(
    geohash_mean
).fillna(global_mean)

# =========================
# LABEL ENCODING
# =========================

categorical_cols = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather"
]

for col in categorical_cols:

    le = LabelEncoder()

    combined = pd.concat(
        [train[col], test[col]],
        axis=0
    )

    le.fit(combined)

    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# =========================
# FEATURES
# =========================

features = [
    "geohash",
    "geohash_freq",
    "geohash_target",
    "day",
    "hour",
    "minute",
    "time_in_minutes",
    "is_peak",
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather"
]

X_train = train[features]
y_train = train["demand"]

X_test = test[features]

# =========================
# MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train)

print("Generating predictions...")
predictions = model.predict(X_test)

# =========================
# SUBMISSION
# =========================

submission = pd.DataFrame({
    "Index": test["Index"],
    "demand": predictions
})

print("\nSubmission Preview:")
print(submission.head())

submission.to_csv("submission.csv", index=False)

print("\nSubmission file created successfully!")
print("Saved as submission.csv")
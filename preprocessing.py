import pandas as pd
import re
import string

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)              # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()         # remove extra spaces
    return text

# -----------------------------
# Load PolitiFact Fake & Real
# -----------------------------
print("Loading dataset...")
df_fake = pd.read_csv("data/politifact_fake.csv")
df_real = pd.read_csv("data/politifact_real.csv")

df_fake["label"] = 0  # fake
df_real["label"] = 1  # real

df = pd.concat([df_fake, df_real], ignore_index=True)

# -----------------------------
# Clean the Title Column
# -----------------------------
print("Cleaning text...")
df["clean_title"] = df["title"].apply(clean_text)

# -----------------------------
# Select final useful columns
# -----------------------------
df_final = df[["clean_title", "label"]]

# -----------------------------
# Save cleaned dataset
# -----------------------------
df_final.to_csv("data/cleaned_data.csv", index=False)
print("Saved cleaned_data.csv successfully!")

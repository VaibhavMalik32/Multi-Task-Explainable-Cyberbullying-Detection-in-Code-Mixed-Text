import pandas as pd
import os
os.makedirs("data", exist_ok=True)
print("Starting Preprocessing...")
data_folder = "data"
files = [f for f in os.listdir(data_folder) if 'BullyExplain' in f]
if not files:
    print(f"Could not find the dataset in the '{data_folder}' folder!")
    exit()
filename = files[0]
filepath = os.path.join(data_folder, filename)
print(f"Found dataset: {filepath}")
if filepath.endswith('.csv'):
    df = pd.read_csv(filepath)
else:
    df = pd.read_excel(filepath)
# 1. Clean missing tweets AND missing targets
df = df.dropna(subset=["tweet", "Target"])
# 2. Map Labels
if "Bully_Label" in df.columns:
    df["Bully_Label"] = df["Bully_Label"].replace({
        "Bully": 1,
        "Non-Bully": 0,
        "Non_bully": 0
    })
    df = df[df["Bully_Label"].isin([0, 1])]
# ---------------- NEW: 5 SUPER CLASS GROUPING ----------------
target_mapping = {
    'Religion': 'Religion & Community',
    'Community': 'Religion & Community',
    'Race': 'Religion & Community',
    'Gender': 'Gender & Sexual Orientation',
    'Sexual Oriantation': 'Gender & Sexual Orientation',  
    'Attacking Relatives and Friends': 'Personal & Relatives',   
    'Profession': 'Profession & Organization',
    'Organization': 'Profession & Organization',   
    'Miscellaneous': 'Miscellaneous / Other',
    'Not Applicable': 'Miscellaneous / Other'
}
df['Target'] = df['Target'].replace(target_mapping)
print(f"Grouped targets into 5 Super Classes: {df['Target'].unique().tolist()}")
# 3. Extracting the Rationale words
def extract_toxic_words(text):
    if not isinstance(text, str): 
        return ""
    words = text.split()
    toxic_words = [w.replace('{', '').replace('}', '').strip() for w in words if '{' in w or '}' in w]
    return " ".join(toxic_words)
if "Explaine" in df.columns:
    df['rationale'] = df['Explaine'].apply(extract_toxic_words)
else:
    print("Error: Could not find the 'Explaine' column!")
    exit()
# 4. Save the cleanly formatted data
clean_df = df[["tweet", "Bully_Label", "Target", "rationale"]]
clean_df.to_csv("data/clean_data.csv", index=False)
print(f"Preprocessing complete!")
print(f"Found {len(clean_df[clean_df['rationale'] != ''])} tweets with explicitly marked toxic words.")
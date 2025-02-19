import pandas as pd
import re

# Load the dataset
input_csv = "web_sentiment_analysis.csv"  
output_csv = "cleaned_sentiment_analysis.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# 1. Remove Rows Containing "Error", "No Text", or "No URL Found" 
unwanted_values = ["error", "no text", "no URL found"]

# Remove rows where any column contains an unwanted value (case insensitive)
df = df[~df.apply(lambda row: row.astype(str).str.lower().isin(unwanted_values).any(), axis=1)]

### 2. Split Emotion Column into Separate Columns ###
def extract_emotion(emotion_text, emotion_type):
    pattern = rf"{emotion_type}:\s([\d\.]+)"  
    match = re.search(pattern, emotion_text)
    return float(match.group(1)) if match else 0  

# Create new columns for each emotion
df["Joy"] = df["emotion"].apply(lambda x: extract_emotion(x, "Joy"))
df["Sadness"] = df["emotion"].apply(lambda x: extract_emotion(x, "Sadness"))
df["Anger"] = df["emotion"].apply(lambda x: extract_emotion(x, "Anger"))
df["Fear"] = df["emotion"].apply(lambda x: extract_emotion(x, "Fear"))
df["Disgust"] = df["emotion"].apply(lambda x: extract_emotion(x, "Disgust"))

# Drop the original emotion column
df.drop(columns=["emotion"], inplace=True)

## this did not work:
df["categories"] = df["categories"].str.replace(r"\\", "", regex=True)  # Remove backslashes


df.to_csv(output_csv, index=False)

print(f"✅ Cleaned data saved to: {output_csv}")

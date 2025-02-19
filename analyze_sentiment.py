import pandas as pd
import os
from googlesearch import search
from newspaper import Article
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions, CategoriesOptions, ConceptsOptions, EmotionOptions, EntitiesOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# IBM Watson NLU Credentials
IBM_API_KEY = "YtP5zP7LS3NQXG5M8u4sAaL2tFN19geo--l-SKMUaJpo" #
IBM_URL = "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/4319860b-6c3e-40a9-b8d4-ac532307131a" #

# Set up IBM NLU API
authenticator = IAMAuthenticator(IBM_API_KEY)
nlu = NaturalLanguageUnderstandingV1(
    version="2021-08-01",
    authenticator=authenticator
)
nlu.set_service_url(IBM_URL)

# Batch size - Adjust to control speed (e.g., 10 rows at a time)
BATCH_SIZE = 10  

# File paths
INPUT_CSV = "Dataset_3k.csv"
OUTPUT_CSV = "web_sentiment_analysis.csv"

def find_article_url(title):
    """Find the real URL of an article using Google Search."""
    try:
        search_results = list(search(title, stop=1, pause=2))
        return search_results[0] if search_results else None
    except Exception as e:
        print(f"Google Search Error: {e}")
        return None

def extract_article_text(url):
    """Extract the main text from an article using newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error extracting article text: {e}")
        return None

def analyze_nlu(text):
    """Analyze text using IBM Watson NLU for sentiment, categories, concepts, emotions, and entities."""
    if not text.strip():
        return {
            "sentiment_label": "Neutral",
            "sentiment_score": 0,
            "categories": "None",
            "concepts": "None",
            "emotion": "None",
            "entities": "None"
        }

    try:
        response = nlu.analyze(
            text=text,
            features=Features(
                sentiment=SentimentOptions(),
                categories=CategoriesOptions(limit=3),
                concepts=ConceptsOptions(limit=3),
                emotion=EmotionOptions(),
                entities=EntitiesOptions(limit=5)
            )
        ).get_result()

        sentiment_label = response["sentiment"]["document"]["label"]
        sentiment_score = response["sentiment"]["document"]["score"]

        categories = ", ".join([cat["label"] for cat in response.get("categories", [])])
        concepts = ", ".join([concept["text"] for concept in response.get("concepts", [])])

        emotions = response["emotion"]["document"]["emotion"]
        emotions_summary = f"Joy: {emotions['joy']:.2f}, Sadness: {emotions['sadness']:.2f}, Anger: {emotions['anger']:.2f}, Fear: {emotions['fear']:.2f}, Disgust: {emotions['disgust']:.2f}"

        entities = ", ".join([entity["text"] for entity in response.get("entities", [])])

        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "categories": categories if categories else "None",
            "concepts": concepts if concepts else "None",
            "emotion": emotions_summary if emotions_summary else "None",
            "entities": entities if entities else "None"
        }

    except Exception as e:
        print(f"IBM NLU Error: {e}")
        return {
            "sentiment_label": "Error",
            "sentiment_score": 0,
            "categories": "Error",
            "concepts": "Error",
            "emotion": "Error",
            "entities": "Error"
        }

if __name__ == "__main__":
    # Load input CSV
    df = pd.read_csv(INPUT_CSV)

    # Check if 'title' column exists
    if "title" not in df.columns:
        print("Error: CSV must contain a 'title' column!")
        exit(1)

    # Ensure there is data to process
    if df.empty:
        print("Error: The dataset is empty!")
        exit(1)

    # Load existing results if the output file already exists (to resume from last processed batch)
    if os.path.exists(OUTPUT_CSV):
        processed_df = pd.read_csv(OUTPUT_CSV)
        processed_titles = set(processed_df["title"])  # Track already processed titles
    else:
        processed_titles = set()

    results = []

    # Process data in batches
    for index, row in df.iterrows():
        title = row["title"]

        # Skip if this title has already been processed
        if title in processed_titles:
            print(f"Skipping (already processed): {title}")
            continue

        print(f"\nSearching for article: {title}")

        # Step 1: Find article URL
        url = find_article_url(title)

        if url:
            print(f"Found URL: {url}")

            # Step 2: Extract article text
            article_text = extract_article_text(url)

            if article_text:
                # Step 3: Analyze using IBM Watson NLU
                nlu_results = analyze_nlu(article_text)
            else:
                nlu_results = {
                    "sentiment_label": "No Text",
                    "sentiment_score": 0,
                    "categories": "No Text",
                    "concepts": "No Text",
                    "emotion": "No Text",
                    "entities": "No Text"
                }
        else:
            print(f"⚠️ No URL found for: {title}")
            nlu_results = {
                "sentiment_label": "No URL Found",
                "sentiment_score": 0,
                "categories": "No URL",
                "concepts": "No URL",
                "emotion": "No URL",
                "entities": "No URL"
            }

        # Append results to list
        results.append({
            "title": title,
            "sentiment_label": nlu_results["sentiment_label"],
            "sentiment_score": nlu_results["sentiment_score"],
            "categories": nlu_results["categories"],
            "concepts": nlu_results["concepts"],
            "emotion": nlu_results["emotion"],
            "entities": nlu_results["entities"]
        })

        # Save results every `BATCH_SIZE` rows
        if len(results) >= BATCH_SIZE:
            batch_df = pd.DataFrame(results)

            if os.path.exists(OUTPUT_CSV):
                batch_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)  # Append without header
            else:
                batch_df.to_csv(OUTPUT_CSV, index=False)  # Create new file

            print(f"Saved batch of {len(results)} rows to: {OUTPUT_CSV}")
            results = []  # Reset batch list

    # Save any remaining results
    if results:
        batch_df = pd.DataFrame(results)
        batch_df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
        print(f"Final batch saved to: {OUTPUT_CSV}")

    print("\n🎉 Batch processing completed!")

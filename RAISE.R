# Load necessary libraries
library(dplyr)
library(stringr)

# ✅ Load dataset
ai_data <- read.csv("cleaned_sentiment_analysis.csv", stringsAsFactors = FALSE)

# ✅ Function to group categories into broader themes
categorize_ai_topic <- function(category) {
  if (str_detect(category, "technology|computing|software|AI|machine learning|automation")) {
    return("Technology & AI")
  } else if (str_detect(category, "business|finance|investing|economy|startups")) {
    return("Business & Economy")
  } else if (str_detect(category, "health|medicine|biotech|pharmaceutical")) {
    return("Healthcare & Biotech")
  } else if (str_detect(category, "education|learning|training|school")) {
    return("Education & Research")
  } else if (str_detect(category, "law|government|politics|policy|regulation")) {
    return("Law & Policy")
  } else if (str_detect(category, "science|space|research|physics|engineering")) {
    return("Science & Innovation")
  } else if (str_detect(category, "media|social media|entertainment|art|culture")) {
    return("Media & Culture")
  } else {
    return("Other")
  }
}

# ✅ Apply categorization function to 'categories' column
ai_data$category_group <- sapply(ai_data$categories, categorize_ai_topic)

# ✅ Summarize sentiment for broader categories
category_sentiment <- ai_data %>%
  group_by(category_group) %>%
  summarise(
    avg_sentiment = mean(sentiment_score, na.rm = TRUE),
    avg_joy = mean(Joy, na.rm = TRUE),
    avg_sadness = mean(Sadness, na.rm = TRUE),
    avg_anger = mean(Anger, na.rm = TRUE),
    avg_fear = mean(Fear, na.rm = TRUE),
    avg_disgust = mean(Disgust, na.rm = TRUE),
    count = n()
  ) %>%
  arrange(desc(avg_sentiment))

print("🔍 Concise AI Category Sentiment Summary:")
print(category_sentiment)




# Load necessary libraries
install.packages("reshape2")
library(reshape2)
library(dplyr)
library(ggplot2)

# ✅ Identify categories with the highest & lowest sentiment
print("🔹 Categories with Highest Sentiment Scores:")
print(category_sentiment %>% arrange(desc(avg_sentiment)) %>% head(3))

print("🔹 Categories with Lowest Sentiment Scores:")
print(category_sentiment %>% arrange(avg_sentiment) %>% head(3))

# ✅ Identify categories with the strongest emotional response
print("🔹 Categories with Highest Joy Scores:")
print(category_sentiment %>% arrange(desc(avg_joy)) %>% head(3))

print("🔹 Categories with Highest Fear Scores:")
print(category_sentiment %>% arrange(desc(avg_fear)) %>% head(3))

print("🔹 Categories with Highest Anger Scores:")
print(category_sentiment %>% arrange(desc(avg_anger)) %>% head(3))

# ✅ Visualize Sentiment Scores Across AI Categories
ggplot(category_sentiment, aes(x = reorder(category_group, avg_sentiment), y = avg_sentiment, fill = category_group)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  labs(title = "Average Sentiment Score by AI Category",
       x = "AI Category",
       y = "Sentiment Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ✅ Visualize Emotional Response to AI Categories
emotion_melted <- reshape2::melt(category_sentiment, id.vars = "category_group", measure.vars = c("avg_joy", "avg_sadness", "avg_anger", "avg_fear", "avg_disgust"))

ggplot(emotion_melted, aes(x = category_group, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Emotional Response by AI Category",
       x = "AI Category",
       y = "Emotion Intensity",
       fill = "Emotion Type") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))




install.packages("corrplot")
library(corrplot)
install.packages("factoextra")
library(factoextra)



# 🔹 1. Correlation Analysis: How are sentiment and emotions related?
emotion_correlation <- ai_data %>%
  select(sentiment_score, Joy, Sadness, Anger, Fear, Disgust) %>%
  cor(use = "pairwise.complete.obs")

print("🔍 Correlation between Sentiment and Emotions:")
print(emotion_correlation)

# ✅ Visualize correlation matrix
corrplot(emotion_correlation, method = "color", type = "lower", tl.cex = 0.8, title = "Correlation Between Sentiment & Emotions")

# 🔹 2. Cluster AI Categories Based on Emotional Patterns
# Normalize data for clustering
emotion_data <- ai_data %>%
  select(Joy, Sadness, Anger, Fear, Disgust) %>%
  scale()

# Use K-Means clustering to group similar sentiment categories
set.seed(123)  # For reproducibility
clusters <- kmeans(emotion_data, centers = 3)  # Adjust cluster number as needed

# Add cluster labels to dataset
ai_data$cluster <- as.factor(clusters$cluster)

# ✅ Visualize clusters using PCA
fviz_cluster(list(data = emotion_data, cluster = clusters$cluster), geom = "point", ellipse.type = "convex", ggtheme = theme_minimal())

# 🔹 3. Sentiment Distribution by AI Category
ggplot(ai_data, aes(x = category_group, y = sentiment_score, fill = category_group)) +
  geom_boxplot() +
  labs(title = "Sentiment Score Distribution by AI Category",
       x = "AI Category",
       y = "Sentiment Score") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 🔹 4. Predictive Modeling: Can emotions predict sentiment?
model <- lm(sentiment_score ~ Joy + Sadness + Anger + Fear + Disgust, data = ai_data)

print("📈 Predictive Model Summary:")
summary(model)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (Update with your actual CSV path)
data = pd.read_csv('reviews.csv')  # Use your own file path here

# Preprocessing: Define sentiment based on the rating
data['sentiment'] = data['rating'].apply(lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')

# Encode the sentiment labels into numeric values
label_encoder = LabelEncoder()
data['sentiment_encoded'] = label_encoder.fit_transform(data['sentiment'])

# Check class distribution before filtering rare classes
print("Class distribution before filtering rare classes:")
print(data['sentiment'].value_counts())

# Remove rare classes (those with fewer than 2 samples)
y_counts = data['sentiment'].value_counts()
rare_classes = y_counts[y_counts < 2].index
data_filtered = data[~data['sentiment'].isin(rare_classes)]

# Re-split the data into features and labels after filtering rare classes
X_filtered = data_filtered['review']
y_filtered = data_filtered['sentiment_encoded']

# Split into training and testing sets (no stratification to avoid the error)
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_train)

# Define class weights for the RandomForestClassifier (using numeric labels)
class_weights = {0: 1, 1: 10, 2: 5}  # Adjust these weights based on the class distribution

# Train a Random Forest model with class weights
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight=class_weights)
rf_model.fit(X_train_balanced, y_train_balanced)

# Predictions and evaluation on the test set
y_pred_rf = rf_model.predict(X_test_tfidf)

# Decode predictions back to original sentiment labels
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred_rf)

# Generate classification report with zero_division handling
rf_classification_report = classification_report(y_test_decoded, y_pred_decoded, zero_division=1)

# Display the classification report for the dataset
print("Classification Report for CSV Data:")
print(rf_classification_report)

# Example reviews
example_reviews = [
    "I love this product, it's amazing!",
    "Terrible, would not recommend.",
    "Very good.",
    "Great value for the price.",
    "Worst purchase ever."
]

# Convert example reviews into a DataFrame
example_df = pd.DataFrame({'review': example_reviews})

# TF-IDF transform the example reviews
example_tfidf = tfidf.transform(example_df['review'])

# Predict sentiment for example reviews
example_predictions = rf_model.predict(example_tfidf)

# Print the sentiment prediction for each example review
print("\nSentiment Predictions for Example Reviews:")
for review, prediction in zip(example_reviews, example_predictions):
    print(f"Review: {review} => Predicted Sentiment: {label_encoder.inverse_transform([prediction])[0]}")

# Plotting sentiment predictions for example reviews
example_df['predicted_sentiment'] = label_encoder.inverse_transform(example_predictions)

plt.figure(figsize=(8, 6))
sns.barplot(x=example_df['predicted_sentiment'].value_counts().index, 
            y=example_df['predicted_sentiment'].value_counts().values, palette='Set2')
plt.title("Sentiment Analysis of Example Reviews", fontsize=16)
plt.xlabel("Sentiment", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

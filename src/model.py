import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, confusion_matrix
import seaborn as sns
import pickle

# Load datasets
tweets = pd.read_csv('./data/Twitter_Data.csv', encoding='ISO-8859-1')
reddit = pd.read_csv('./data/Reddit_Data.csv', encoding='ISO-8859-1')

# Combine datasets
combined_data = pd.concat([tweets, reddit], ignore_index=True)

# Display dataset info and check for missing values
print(combined_data.info())
print("Missing: ", combined_data.isnull().sum())

# Drop missing values
combined_data = combined_data.dropna()
print("Missing after dropping: ", combined_data.isnull().sum())

# Plot sentiment distribution
combined_data['sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Analysis of Combined Data')
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1, 2], labels=['Negative', 'Neutral', 'Positive'])
plt.savefig("./../graphs/bar_combined.png")

# Define the number of samples for each sentiment
numero_resultados = 35000

# Sample an equal number of positive, negative, and neutral sentiments
resultados_positivos = combined_data[combined_data['sentiment'] == 1].sample(n=numero_resultados, random_state=23)
resultados_negativos = combined_data[combined_data['sentiment'] == -1].sample(n=numero_resultados, random_state=23)
resultados_neutros = combined_data[combined_data['sentiment'] == 0].sample(n=numero_resultados, random_state=23)

# Combine the sampled data
final_data = pd.concat([resultados_positivos, resultados_negativos, resultados_neutros])
print(final_data['sentiment'].value_counts())

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train, X_test, y_train, y_test = train_test_split(final_data['text'], final_data['sentiment'], test_size=0.2, random_state=23)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Save the TF-IDF vectorizer for later use
with open('./saved_models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Convert sparse matrix to dense matrix
X_train_vectorized = X_train_vectorized.todense()
X_test_vectorized = X_test_vectorized.todense()

# Encode labels
encoder = LabelEncoder()
y_train_encoded = to_categorical(encoder.fit_transform(y_train))
y_test_encoded = to_categorical(encoder.transform(y_test))

# Define neural network model
model = Sequential()
model.add(Dense(512, input_shape=(X_train_vectorized.shape[1],), activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_vectorized, y_train_encoded, epochs=10, batch_size=128,
                    validation_data=(X_test_vectorized, y_test_encoded), verbose=1)

# Save the model
model.save('./saved_models/sentiment_model.h5')

# Plot model accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("./../graphs/precision_combined.png")

# Predict and evaluate the model
y_pred = model.predict(X_test_vectorized)
y_pred_classes = y_pred.argmax(axis=1)
y_test_classes = y_test_encoded.argmax(axis=1)

# Calculate precision
precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
print(f'Weighted Precision: {precision}')

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("./../graphs/confusion_matrix.png")

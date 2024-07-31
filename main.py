import os
import glob

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




positive_path = "mldata/pos/"
negative_path = "mldata/neg/"

positive_files = os.listdir(positive_path)
negative_files = os.listdir(negative_path)

def clean_text(text):
  text = text.lower()
  text = ''.join([char for char in text if char.isalnum() or ' ' in char])
  tokens = word_tokenize(text)
  stop_words = stopwords.words('english')
  tokens = [word for word in tokens if word not in stop_words]
  return " ".join(tokens)

def read_review(path, filelist , label ):
  reviews = []
  for filename in filelist:
    full_path = os.path.join(path, filename)
    with open(full_path, 'r', encoding= 'utf-8') as file:
      review = file.read().strip()
      reviews.append((review, label))
  return reviews


positive_reviews = read_review(positive_path, positive_files, "positive")
negative_reviews = read_review(negative_path, negative_files, "negative")

all_reviews = positive_reviews + negative_reviews

all_reviews = [ (clean_text(review), label) for review, label in all_reviews  ]

print(len(all_reviews))
print(len(positive_reviews))
print(len(negative_reviews))

vectorizer = CountVectorizer(max_features= 2000)
filtered_reviews = [(review, label) for review, label in all_reviews if label is not None]
review_text = [review for review, _ in filtered_reviews]

print(len(filtered_reviews))

vectorizer.fit(review_text)
review_features = vectorizer.transform([review for review, _ in all_reviews])
review_train, review_test, label_train, label_test = train_test_split([review for review, _ in all_reviews], [label for _ ,  label in all_reviews], test_size = 0.2)


model = MultinomialNB()
model.fit(review_features, label_train)

print("Hello World")
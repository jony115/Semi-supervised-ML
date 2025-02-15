# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:13:03 2025

@author: Jony

"""

# Import Datset into Python 
import pandas as pd 
# Load the Excel file 
data = pd.read_excel("F:\Msc BA\Sem 2\Data mining\Assignment 2\A_II_Emotion_Data_Student_Copy_Final.xlsx") 
# Display the first few rows of the DataFrame 
print(data.head()) 
print(data['text_reviews_']) 
# rename and store variables  
df = pd.DataFrame(data['text_reviews_']) 
emotions = data['emotions_'].unique() 
# Display the first few rows of the DataFrame 
print(data) 
print(data.head()) 
print("Different emotions:", emotions) 

# TEXT CLEANING 
# Import all the libraries for text cleaning  
import pandas as pd 
import re 
import string 
import nltk 

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer 
# Download necessary NLTK resources 
nltk.download('punkt') 
nltk.download('stopwords') 
nltk.download('wordnet') 
# Initialize the Porter Stemmer and WordNetLemmatizer 
stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer() 
def clean_text(text): 
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags 
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation 
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-word characters 
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces 
    text = text.lower()  # Convert text to lowercase 
    return text 
def remove_stopwords(text):    # Remove stopwords and recombine text 
    tokens = word_tokenize(text) 
    stop_words = set(stopwords.words('english')) 
    return ' '.join([word for word in tokens if word.lower() not in stop_words]) 
def stem_text(text):
     # Apply stemming to each token and recombine into a single string 
    tokens = word_tokenize(text) 
    return ' '.join([stemmer.stem(word) for word in tokens]) 
def lemmatize_text(text): # Apply lemmatization to each token and recombine into a single string 
    tokens = word_tokenize(text) 
    tokens = [lemmatizer.lemmatize(word) for word in tokens] 
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens]) 
# Apply all functions in sequence 
data['cleaned_text'] = data['text_reviews_'].apply(clean_text) 
data['filtered_data'] = data['cleaned_text'].apply(remove_stopwords) 
data['stemmed_data'] = data['filtered_data'].apply(stem_text) 
data['review'] = data['stemmed_data'].apply(lemmatize_text) 
review = data['review'] 
# Display the final DataFrame 
# Print DataFrames 
print("Cleaned data:", review) 
print("emotions:", emotions) 

## MODEL BUILDING 
# Import all the packages needed for building the model

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.svm import SVC 
from sklearn.metrics import f1_score 
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import FunctionTransformer 
from sklearn.semi_supervised import SelfTrainingClassifier 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# Handling self-training with label and unlabelled data 
# Split data into labeled and unlabeled based on 'emotions_' column 
unlabeled_data = data[data['emotions_'] == 'NaN'][['review']] 
unlabeled_data['emotions_'] = -1 
print("Unlabeled data preview:", unlabeled_data) 

# Define labeled data where "Emotions" is not missing or 'NaN' 
labeled_data = data[data['emotions_'].notna() & (data['emotions_'] != 'NaN')] 
y_labeled = labeled_data['emotions_'] 
y_unlabeled = unlabeled_data['emotions_'] 
X_labeled = labeled_data['review'] 
X_unlabeled = unlabeled_data['review'] 
print("Labeled Y:", y_labeled) 
print("Unlabeled Y:", y_unlabeled) 
print("Unlabeled X:", X_unlabeled) 
print("Labeled X:", X_labeled) 

# Define SVM and Vectorizer parameters 
svm_params = {'C': 1.0, 'kernel': 'linear', 'gamma': 'auto', 'probability': True} 
vectorizer_params = {'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.8} 
# Create a supervised SVM Pipeline 
svm_pipeline = Pipeline([ 
    ("vect", CountVectorizer(**vectorizer_params)), 
    ("tfidf", TfidfTransformer()), 
    ("clf", SVC(**svm_params)), 
]) 

# Create a SelfTraining Pipeline using the same SVM model 
st_pipeline = Pipeline([ 
    ("vect", CountVectorizer(**vectorizer_params)), 
    ("tfidf", TfidfTransformer()), 
    ("clf", SelfTrainingClassifier(SVC(**svm_params), verbose=True)), 
]) 

# Function to evaluate and print metrics of the classifier 
def eval_and_print_metrics(clf, X_train, y_train, X_test, y_test): 
    print("Number of training samples:", len(X_train)) 
    print("Unlabeled samples in training set:", sum(1 for x in y_train if x == -1)) 
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test) 
    print("Train Labels:", y_train) 
    print("Predicted Labels:", y_pred) 
    print("Test Labels:", y_test) 
    print("Micro-averaged F1 score on test set:", f1_score(y_test, y_pred, average="micro")) 
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1)) 
    # Generate and print the confusion matrix 
    cm = confusion_matrix(y_test, y_pred) 
    print("Confusion Matrix:\n", cm) 
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1)) 
    # Visualization of the Confusion Matrix using Heatmap 
    plt.figure(figsize=(8, 6)) 
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues") 
    plt.title('Confusion Matrix Heatmap') 
    plt.xlabel('Predicted Labels') 
    plt.ylabel('True Labels') 
    plt.show() 

# Split the labeled data for training and testing 
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, stratify=y_labeled, 
random_state=42) 
# Print results 
print("Evaluating Supervised SVM Classifier on the labeled data:") 
eval_and_print_metrics(svm_pipeline, X_train, y_train, X_test, y_test) 
# Managing Labeled and Unlabeled Data for Self Training 
test_indices = X_test.index 
print("Test indices:", test_indices) 
# Exclude test data from labeled data for self-training 
X_labeled_filtered = X_labeled.drop(index=test_indices, errors='ignore') 
y_labeled_filtered = y_labeled.drop(index=test_indices, errors='ignore') 
# Concatenate filtered labeled data with unlabeled data for training 
X = pd.concat([X_labeled_filtered, X_unlabeled]) 
y = pd.concat([y_labeled_filtered, y_unlabeled]) 

# Define and apply a mapping for seven different emotions including unlabeled data 
label_mapping = { 
'joy': 0, 'fear': 1, 'surprise': 2, 'anger': 3, 'neutral': 4, 'disgust': 5, 'sadness': 6, -1: -1 
} 
y = [label_mapping[label] for label in y] 
y_test = [label_mapping[label] for label in y_test] 
print("Mapped Test Labels:", y_test) 
print("Self Training Classifier on the labeled and unlabeled data:") 
eval_and_print_metrics(st_pipeline, X, y, X_test, y_test) 
# Supervised ML - Random forest 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import TfidfVectorizer 
X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42) 
# Text preprocessing and conversion to a matrix of TF-IDF features 
tfidf_vectorizer = TfidfVectorizer() 
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train) 
X_test_tfidf = tfidf_vectorizer.transform(X_test) 
# random forest classifier 
clf_RFC = RandomForestClassifier(n_estimators=100, random_state=42) 
clf_RFC.fit(X_train_tfidf, y_train) 
y_pred = clf_RFC.predict(X_test_tfidf) # Predict on the test set 
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
# Visualization of the Confusion Matrix using Heatmap 
plt.figure(figsize=(8, 6)) 
cm = confusion_matrix(y_test, y_pred) 
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds") 
plt.title('Confusion Matrix Heatmap') 
plt.xlabel('Predicted Labels') 
plt.ylabel('True Labels') 
plt.show() 
## VISUALIZATIONS 
# Bar chart of Star rating based on emotions 
# Calculate the average star ratings for each emotions 
average_ratings = data.groupby('emotions_')['star_rating_'].mean().reset_index() 
# Print average ratings 
print(average_ratings) 
# Create a seaborn bar plot 
ax = sns.barplot(x='emotions_', y='star_rating_', data=average_ratings, palette='crest') 
plt.title('Comparison of Average Star Ratings of emotions')

plt.ylim(0, 5)  # Star ratings are usually out of 5 

# Annotate each bar with the average rating 
for p in ax.patches: 
    ax.annotate(format(p.get_height(), '.1f'),  # Format the number to one decimal place 
    (p.get_x() + p.get_width() / 2., p.get_height()),  # Position for the annotation 
    ha = 'center',  # Center the text horizontally 
    va = 'center',  # Center the text vertically within the bar 
    xytext = (0, 9),  # Distance from the top of the bar 
    textcoords = 'offset points') 
    plt.show() 
# Comparision Bar chart of Brands and thier star ratings 
# Calculate the average star ratings for each brand 
average_ratings = data.groupby('brand_name_')['star_rating_'].mean().reset_index() 
# Print average ratings 
print(average_ratings) 
# Create a seaborn bar plot 
ax = sns.barplot(x='brand_name_', y='star_rating_', data=average_ratings, palette='mako') 
plt.title('Comparison of Average Star Ratings Between Two Brands') 
plt.ylim(0, 5)  # Star ratings are usually out of 5 

# Annotate each bar with the average rating 
for p in ax.patches: 
    ax.annotate(format(p.get_height(), '.1f'),  # Format the number to one decimal place 
    (p.get_x() + p.get_width() / 2., p.get_height()),  # Position for the annotation 
    ha = 'center',  # Center the text horizontally 
    va = 'center',  # Center the text vertically within the bar 
    xytext = (0, 9),  # Distance from the top of the bar 
    textcoords = 'offset points') 
    plt.show() 
    
# Number of companies per country 
company_count = data['country_'].value_counts().reset_index() 
company_count.columns = ['Country', 'Number of Companies'] 
# Plotting 
plt.figure(figsize=(15, 6)) 
sns.barplot(x='Country', y='Number of Companies', data=company_count, color='#072640') 
plt.title('Number of Companies per Country') 
plt.xlabel('Country') 
plt.ylabel('Number of Companies') 
plt.xticks(rotation=90, fontsize='small')  
plt.show() 























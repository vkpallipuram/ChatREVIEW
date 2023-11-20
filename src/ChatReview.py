#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nrclex import NRCLex
import pprint
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import operator
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored
import openai

openai.api_key = ""
REVIEWS_FILE_PATH = '../data/Hospitality_Data/HospitalityReviews.txt'
CATEGORIES_QUESTIONS_FILEPATH = '../data/Hospitality_Data/hospitallity_explicit.csv'
# Education: ["negative", "anger", "sadness"]
# Restaurant: ["negative", "anger", "disgust"]
# Hospitality: ["negative", "anger", "fear", "disgust"]
TARGET_EMOTIONS = ["negative", "anger", "fear", "disgust"]
WORD_CLOUD_PNG ="../OUTPUT/wordcloud_hospitality.png"
OUTPUT_FILEPATH = "../OUTPUT/output_hospitality.txt"


# In[2]:


# Function to read and clean dataset
def read_and_clean_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = f.read()

    # Remove punctuation, numbers, convert to lowercase
    dataset = re.sub(r'[^\w\s]', '', dataset)
    dataset = re.sub(r'\d+', '', dataset)
    dataset = dataset.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = dataset.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)


# In[3]:


# Function to count word frequencies
def count_word_frequencies(dataset):
    counter = Counter(dataset.split())
    return counter


# In[4]:


# Function to analyze emotions in text
def analyze_emotions(dataset):
    text_object = NRCLex(dataset)
    return text_object.affect_dict


# In[5]:


# Function to generate word cloud
def generate_word_cloud(frequencies, file_name=WORD_CLOUD_PNG):
    wordcloud = WordCloud(width=600, height=300, background_color='white', max_font_size=60)
    wordcloud.generate_from_frequencies(frequencies)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('Words Associated with ' + ", ".join(TARGET_EMOTIONS))
    plt.axis('off')
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.show()
    


# In[6]:


def words_with_emotions(emotion_dict, target_emotions):
    combined_word_frequencies = {}

    for emotion in target_emotions:
        words_with_emotions = [word for word, emotions in emotion_dict.items() if emotion in emotions]
        combined_word_frequencies[emotion] = words_with_emotions
    
    return combined_word_frequencies 


# In[7]:


# Function to process words by emotions
def process_words_by_emotions(combined_word_frequencies, word_frequencies):
    unique_words = set(word for category in combined_word_frequencies.values() for word in category)
    # Associating each unique word with its count
    word_frequencies = {word: word_frequencies[word] for word in unique_words}
    return word_frequencies


# In[8]:


def main():
    with open(OUTPUT_FILEPATH, 'w') as file:
        pp = pprint.PrettyPrinter(indent=4, stream=file)

        # Read and clean dataset
        cleaned_dataset = read_and_clean_dataset(REVIEWS_FILE_PATH)

        # Analyze emotions in the dataset
        emotion_dict = analyze_emotions(cleaned_dataset) # return a dict
        print("Emotion Dictionary (Words and their associated emotions):", file=file)
        pp.pprint(emotion_dict)  # Debug print

        # Count word frequencies
        word_frequencies = count_word_frequencies(cleaned_dataset) # return a collections.Counter
    #     print("Word Frequencies:")
    #     pp.pprint(word_frequencies)  # Debug print

        combined_word_freqs = words_with_emotions(emotion_dict, TARGET_EMOTIONS)
        print("\nTarget emotions and their associated keywords (target keywords):", file=file)
        pp.pprint(combined_word_freqs)
        r_emo = process_words_by_emotions(combined_word_freqs, word_frequencies)
        print("\nFrequency of Target Keywords: ", file=file)
        pp.pprint(r_emo)  # Debug print

        # Generate word cloud for multiple emotions
        if combined_word_freqs:  # Check if combined_word_freqs is not empty
            generate_word_cloud(r_emo)
            print("\n.....Generated Word Cloud: ", WORD_CLOUD_PNG, file=file)
        else:
            print("No words found for the specified emotions.", file=file)  # Debug message

        # Read the csv file
        df = pd.read_csv(CATEGORIES_QUESTIONS_FILEPATH)

        # Get the 'Category' column and convert it to a list
        categories = df['Category'].tolist()
    #     pp.pprint(categories)
        keywords = list(r_emo.keys())             
    #     print(keywords)

        # Initialize the model (it will download the parameters the first time)
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create vectors for categories and keywords
        category_vectors = {category: model.encode(category) for category in categories}
        keyword_vectors = {keyword: model.encode(keyword) for keyword in keywords}

        # Create a dictionary to hold the grouped keywords for each category
        grouped_keywords = {category: [] for category in categories}

        for category, category_vector in category_vectors.items():
            for keyword, keyword_vector in keyword_vectors.items():
                similarity = cosine_similarity([category_vector], [keyword_vector])[0][0]
                if similarity > 0.25:  # you may want to adjust this threshold
                    grouped_keywords[category].append(keyword)

        # Convert grouped keywords into a list of tuples (category, keyword count)
        keyword_counts = [(category, len(keywords)) for category, keywords in grouped_keywords.items()]

        # Sort by keyword count in descending order
        keyword_counts.sort(key=lambda x: x[1], reverse=True)

        # Use one for loop to print Category, Keyword count, and Keywords
        print("\nCategories:", file=file)
        for category, count in keyword_counts:
            # Retrieving keywords for the current category
            keywords = grouped_keywords[category]
            print(f"Category: {category}\n\tKeyword count: {count}\n\tKeywords: {keywords}\n", file=file)

        # Create a dictionary for easier lookup
        question_dict = pd.Series(df.Question.values,index=df.Category).to_dict()

        # Print the top 3 categories with their corresponding questions
        cat = []
        print("\nTop 3 categories with their corresponding questions:", file=file)
        for category, count in keyword_counts[:3]:
            print(f"Category: {category}\n\tKeyword count: {count}\n\tQuestion: {question_dict[category]}\n", file=file)
            cat.append(category)
    #     pp.pprint(cat) # print top 3 categories

        # Get the top 3 categories with their corresponding questions
        questions = [question_dict[category] for category, _ in keyword_counts[:3]]

        print("\nGPT Generated Recommendations based on Questions:", file = file)
        # Use each question as input to the GPT-3.5-turbo model
        for question in questions:
            print(f"Question: {question}\n", file=file)

            output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": question}
                ],
                temperature=0,
                stream=True
            )

            # Iterate through the chunks and print the content
            for chunk in output:
                if 'content' in chunk['choices'][0]['delta']:
#                     print(colored(chunk['choices'][0]['delta']['content'], 'blue'), end = "", file=file)
                     print(chunk['choices'][0]['delta']['content'], end = "", file = file)

            print("\n", file=file)


main()


# In[ ]:





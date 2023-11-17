#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nrclex import NRCLex
import pprint

pp = pprint.PrettyPrinter(indent=4)

with open('./Restaurant_Data/OGI_Restaurant_40.txt', 'r', encoding='utf-8') as f:
    reviews = f.readlines()

reviews = ';'.join(reviews)
text_object = NRCLex(reviews)

original_dict = text_object.affect_dict
result_dict = {}

for word, emotions in original_dict.items():
    for emotion in emotions:
        if emotion not in result_dict:
            result_dict[emotion] = []
        result_dict[emotion].append(word)


# In[2]:


from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import operator
from pprint import pprint

# Read the text file
with open('./Restaurant_Data/OGI_Restaurant_40.txt', 'r', encoding='utf-8') as f:
    dataset = f.read()

# Clean the dataset
cleaned_dataset = re.sub(r'[^\w\s]', '', dataset)  # Remove punctuation
cleaned_dataset = re.sub(r'\d+', '', cleaned_dataset)  # Remove numbers
cleaned_dataset = cleaned_dataset.lower()  # Convert to lowercase

# Remove stop words
stop_words = set(stopwords.words('english'))
words = cleaned_dataset.split()
filtered_words = [word for word in words if word not in stop_words]

# Join the filtered words back into a string
cleaned_dataset = ' '.join(filtered_words)

# Print the cleaned dataset
# print(cleaned_dataset)

counter = {}

# Split the string into words
words = cleaned_dataset.split()

# Count the occurrences of each word
for word in words:
    if word not in counter:
        counter[word] = 0
    counter[word] += 1

pprint(counter) #dict


# In[3]:


combined_list = []
def WordsByEmotions(emotionStrs):
    # Dictionary to store individual word frequencies for each emotion
    individual_word_frequencies = {emotion: {} for emotion in emotionStrs}

    # Create a dictionary to store combined frequencies of words across emotions
    combined_word_frequencies = {}

    # Iterate over all given emotions
    for emotionStr in emotionStrs:
        emotion_words = result_dict.get(emotionStr, [])
        
        for word in emotion_words:
            # Check if the word exists in the counter dictionary
            if word in counter:
                # For individual word frequencies per emotion
                if word not in individual_word_frequencies[emotionStr]:
                    individual_word_frequencies[emotionStr][word] = 0
                individual_word_frequencies[emotionStr][word] += counter[word]

                # Add the frequency of the word to the combined_word_frequencies dictionary
                if word not in combined_word_frequencies:
                    combined_word_frequencies[word] = 0
                combined_word_frequencies[word] += counter[word]

    # Print individual word frequencies for each emotion
    for emotion, word_freqs in individual_word_frequencies.items():
        sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
        print(f"Word Frequencies for {emotion}:")
        for word, freq in sorted_word_freqs:
            print(f"{word}: {freq}")
        print("\n")

    # Print sorted combined word frequencies
    sorted_combined_word_freqs = sorted(combined_word_frequencies.items(), key=operator.itemgetter(1), reverse=True)
    print("Combined Word Frequencies:")
    for word, freq in sorted_combined_word_freqs:
        combined_list.append(word)
        print(f"{word}: {freq}")
    print("\n")

    # Create the word cloud from the combined frequencies
    wordcloud = WordCloud(width=600, height=300, background_color='white', max_font_size=60)
    wordcloud.generate_from_frequencies(combined_word_frequencies)
    
    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    title = 'Words Associated with ' + ", ".join(emotionStrs)
    plt.title(title)
    plt.axis('off')
    plt.savefig("wordcloud.png", dpi=300, bbox_inches='tight')
    plt.show()


# Generate word cloud for multiple emotions and print sorted counts
WordsByEmotions(["negative", "anger", "disgust"])
# WordsByEmotion("negative")
# WordsByEmotion("anger")
# WordsByEmotion("sadness")


# In[4]:


import pandas as pd

# Read the csv file
df = pd.read_csv('./Restaurant_Data/paper_user_restaurant_explicit.csv')

# Get the 'Categories' column and convert it to a list
categories = df['Category'].tolist()
keywords = combined_list
print(combined_list)


# In[5]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the BERT model (it will download the parameters the first time)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create vectors for categories and keywords
category_vectors = {category: model.encode(category) for category in categories}
keyword_vectors = {keyword: model.encode(keyword) for keyword in keywords}

# Create a dictionary to hold the grouped keywords for each category
grouped_keywords = {category: [] for category in categories}

for category, category_vector in category_vectors.items():
    for keyword, keyword_vector in keyword_vectors.items():
        similarity = cosine_similarity([category_vector], [keyword_vector])[0][0]
        if similarity > 0.25:  # we may want to adjust this threshold
            grouped_keywords[category].append(keyword)

# Print grouped keywords
for category, keywords in grouped_keywords.items():
    print(f"Category:", f"{category}\nKeywords: {keywords}\n")


# In[6]:


# Convert grouped keywords into a list of tuples (category, keyword count)
keyword_counts = [(category, len(keywords)) for category, keywords in grouped_keywords.items()]

# Sort by keyword count in descending order
keyword_counts.sort(key=lambda x: x[1], reverse=True)

# Print the top 3 categories
cat = []
for category, count in keyword_counts[:10]:
    print(f"Category: {category}\nKeyword count: {count}\n")
    cat.append(category)
    
print(cat)
    


# ### Organization (for each of the categories, pre-assigned questions)
# --> 26 questions
# Next meeting: run Manohar's data --> word clouds --> categories --> questions to GPT --> Result from ChatGPT
# 
# Present: in action, randomly select 50/150/180 reviews from database (after expanded)
# 2 demonstrations: 1/ How the code works 2/general public/audience presentation (storyline/powerpoint)

# In[7]:


# Create a dictionary for easier lookup
question_dict = pd.Series(df.Question.values,index=df.Category).to_dict()

# Print the top 3 categories with their corresponding questions
for category, count in keyword_counts[:3]:
    print(f"Category: {category}\nKeyword count: {count}\nQuestion: {question_dict[category]}\n")


# In[8]:


import openai
openai.api_key = "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"


# In[9]:


import pandas as pd
import openai
from termcolor import colored
# Get the top 3 categories with their corresponding questions
questions = [question_dict[category] for category, _ in keyword_counts[:3]]

# Use each question as input to the GPT-3.5-turbo model
for question in questions:
    print(f"Question: {question}\n")
    
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
            print(colored(chunk['choices'][0]['delta']['content'], 'blue'), end = "")

    print("\n")


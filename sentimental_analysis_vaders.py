# sentimental_analysis_vaders_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag, chunk
from tqdm import tqdm

# Setup
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

plt.style.use('ggplot')
st.title("📊 Sentiment Analysis with VADER")

# File upload
uploaded_file = st.file_uploader(r"C:\Users\KUSHAGRA\Downloads\Reviews.csv\Reviews.csv", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Sample Data")
    st.write(df.head())

    # Plot: Count of Reviews by Stars
    st.subheader("⭐ Review Score Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    df['Score'].value_counts().sort_index().plot(kind='bar', ax=ax1)
    ax1.set_title('Count of Reviews by Stars')
    ax1.set_xlabel('Review Stars')
    ax1.set_ylabel('Count')
    st.pyplot(fig1)

    # Show example review
    st.subheader("📝 Example Review")
    example = df['Text'].iloc[50]
    st.write(example)

    # Tokenization and POS tagging
    tokens = word_tokenize(example)
    tagged = pos_tag(tokens)
    entities = chunk.ne_chunk(tagged)

    st.subheader("🔍 Named Entity Recognition")
    st.text(entities.pformat())

    # Sentiment Analysis
    st.subheader("💬 Running VADER Sentiment Analysis")
    sia = SentimentIntensityAnalyzer()
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row['Text']
        myid = row['Id']
        res[myid] = sia.polarity_scores(text)

    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'Id'})
    vaders = vaders.merge(df, how='left')

    # Plot: Compound Score
    st.subheader("📈 Compound Sentiment Score by Review Score")
    fig2, ax2 = plt.subplots()
    sns.barplot(data=vaders, x='Score', y='compound', ax=ax2)
    ax2.set_title('Compound Score Analysis')
    st.pyplot(fig2)

    # Plot: Positive, Neutral, Negative Scores
    st.subheader("📊 Sentiment Breakdown")
    fig3, axs = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
    axs[0].set_title('Positive')
    sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
    axs[1].set_title('Neutral')
    sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
    axs[2].set_title('Negative')
    st.pyplot(fig3)

else:
    st.info("Please upload a CSV file with 'Text', 'Score', and 'Id' columns to begin.")
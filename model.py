# model.py

import pandas as pd
import pickle as pk
import nltk
from utility import clean_text

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Pickles
final_rating = pk.load(open('pickle/item_final_rating.pkl', 'rb'))
tfidf = pk.load(open('pickle/tfidf.pkl', 'rb'))
model = pk.load(open('pickle/Logistic_Regression.pkl', 'rb'))

product_df = pd.read_csv('data/sample30_cleand.csv')

import pandas as pd

# ---- Step 1: Get top 20 IBCF products ----

def get_top_20_products(user_id, final_rating, verbose=True):
    if user_id not in final_rating.index:
        if verbose:
            print(f"User '{user_id}' not found in final_rating.")
        return []
    
    top_20_names = final_rating.loc[user_id].sort_values(ascending=False).head(20).index.tolist()
    if verbose:
        print(f"Top 20 product names for user '{user_id}': {top_20_names}")
    return top_20_names

# ---- Step 2: Predict positive sentiment for all reviews ----

def compute_sentiment_probabilities(text_series, sentimodel, tfidf_vectorizer, verbose=True):
    clean_texts = text_series.astype(str)
    clean_texts = clean_texts.apply(lambda x: clean_text(x))
    tfidf_input = tfidf_vectorizer.transform(clean_texts)
    sentiment_probs = sentimodel.predict_proba(tfidf_input)
    pos_probs = sentiment_probs[:, 1]
    if verbose:
        print(f"Computed sentiment probabilities for {len(pos_probs)} texts.")
    return pos_probs

# ---- Step 3: Aggregate average sentiment score ----

def aggregate_sentiment_scores(reviews_df, verbose=True):
    summary = reviews_df.groupby('name')['positive_sentiment_prob'].mean().reset_index()
    summary = summary.rename(columns={'positive_sentiment_prob': 'avg_positive_sentiment'})
    if verbose:
        print(f"Aggregated sentiment scores for {summary.shape[0]} products.")
    return summary

# ---- Step 4: Build top 5 recommendations DataFrame ----

def build_top_5_df(review_df, top_5_names, sentiment_summary, verbose=True):
    top_5_df = review_df[review_df['name'].isin(top_5_names)][['id', 'name', 'brand', 'categories']].drop_duplicates()
    top_5_df = pd.merge(top_5_df, sentiment_summary, on='name', how='left')
    if verbose:
        print(f"Built top 5 DataFrame with {top_5_df.shape[0]} entries.")
    return top_5_df

# ---- Main function: Recommend Top 5 ----

def recommend_top_products(user_id, final_rating, review_df, sentimodel, tfidf_vectorizer, verbose=True):
    # 1. Get top 20 UBCF
    top_20_names = get_top_20_products(user_id, final_rating, verbose=verbose)
    if verbose:
        print(f"Top 20 product names for user '{user_id}': {top_20_names}")
    if not top_20_names:
        return pd.DataFrame(columns=['id', 'name', 'brand', 'categories', 'avg_positive_sentiment'])
    
    # 2. Filter reviews
    candidate_reviews = review_df[review_df['name'].isin(top_20_names)].copy()
    if verbose:
        print(f"Number of candidate reviews for user '{user_id}': {candidate_reviews.shape[0]}")
    # 3. Compute sentiment probabilities
    candidate_reviews['positive_sentiment_prob'] = compute_sentiment_probabilities(
        candidate_reviews['reviews_text'],
        sentimodel,
        tfidf_vectorizer,
        verbose=verbose
    )
    
    # 4. Aggregate sentiment
    sentiment_summary = aggregate_sentiment_scores(candidate_reviews, verbose=verbose)
    
    # 5. Top 5 product IDs
    top_5_names = sentiment_summary.sort_values(by='avg_positive_sentiment', ascending=False).head(5)['name']
    
    # 6. Build top 5 DataFrame
    top_5_df = build_top_5_df(review_df, top_5_names, sentiment_summary, verbose=verbose)
    if verbose:
        print(f"Top 5 recommended products for user '{user_id}': {top_5_df}")
    return top_5_df


def recommend_products(user_id, verbose=True):
    """
    Recommend top 5 products for a given user based on UBCF and sentiment analysis.
    
    Args:
        user_id (str): The user ID for whom to recommend products.
    
    Returns:
        pd.DataFrame: DataFrame containing top 5 recommended products with their details.
    """
    return recommend_top_products(user_id, final_rating, product_df, model, tfidf, verbose=verbose)


def get_user_rated_products(user_id, verbose=True):
    """
    Return all products reviewed by the user, sorted by reviews_rating (descending).
    Args:
        user_id (str): Username
        csv_path (str): Path to CSV
    Returns:
        pd.DataFrame: Products rated by user
    """
    df = product_df
    user_reviews = df[df['reviews_username'] == user_id]
    user_reviews = user_reviews.sort_values(by='reviews_rating', ascending=False)
    if verbose:
        print(user_reviews.shape)
    return user_reviews[['name', 'brand', 'categories', 'reviews_rating', 'reviews_text']]
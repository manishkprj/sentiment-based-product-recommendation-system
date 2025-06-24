# Sentiment-Based Product Recommendation System

**Demo URL**: [https://product-recommendation-130dbb6b216e.herokuapp.com/](https://product-recommendation-130dbb6b216e.herokuapp.com/)

**Source Code**: [https://github.com/manishkprj/sentiment-based-product-recommendation-system/](https://github.com/manishkprj/sentiment-based-product-recommendation-system/)

---

## Problem Statement

The e-commerce market is rapidly evolving, with giants like **Amazon** and **Flipkart** dominating the space. In this landscape, **Ebuss**, an emerging player, must innovate by offering superior product recommendations to retain customers and grow market share.

Our mission is to build a **Sentiment-Based Product Recommendation System** that:

 Understands **user preferences** from their reviews and ratings  
 Enhances recommendations using **sentiment analysis** of reviews  
 Delivers a **user-friendly web interface** to deploy these recommendations  

---

## Project Objectives

 Data sourcing and sentiment analysis  
 Building a recommendation system  
 Improving recommendations with the sentiment analysis model
 Deploying the system with a web app  

---

## Dataset

Inspired by a Kaggle competition and provided as a subset.  
Contains:

- `reviews_username`  
- `name` (product name)  
- `reviews_rating`  
- `reviews_text`  
- `user_sentiment` (Positive / Negative)  
- Product metadata (category, brand, etc.)  

**Assumption**: The number of users and products is fixed — no new users or products will be added.

---

## Pipeline Steps

### Exploratory Data Analysis (EDA)

- Uncovered review patterns  
- Detected missing values  
- Analyzed rating distributions  
- Profiled active users and popular products  

### Data Cleaning

- Handled missing values  
- Removed duplicates  
- Unified text formats  
- Dropped irrelevant columns  

### Text Preprocessing

- Lowercased reviews  
- Removed punctuation and stopwords  
- Applied **lemmatization**  
- Prepared clean text for feature extraction  

### Feature Extraction

- **TF-IDF Vectorizer** with:  
  - `ngram_range = (1,3)`  
  - `max_features = 10000`  

---

## Sentiment Analysis Model

Built and validated **4 models**:

1. **Logistic Regression**  
2. **Random Forest**  
3. **Naive Bayes**  
4. **XGBoost**  

### Model Tuning:

- **SMOTE** to balance classes  
- **Grid Search** for hyperparameter tuning  
- Evaluated on:  
  - Accuracy  
  - Precision  
  - Recall  
  - Specificity  
  - F1 Score  
  - ROC AUC  

### Final Model Selected:

**Logistic Regression**

### Why Logistic Regression?

- Delivered **highest overall balanced performance**  
- Very good **recall** and **specificity**  
- Less prone to overfitting than tree models  
- Generalizes well to unseen users  
- Simple, fast, and interpretable  

---

## Recommendation System

### Two types explored:

1. **User-User Collaborative Filtering**  
2. **Item-Item Collaborative Filtering**  


### Final Selection:

 **Item-Item Collaborative Filtering**

### Why Item-Item?

- **More stable** than user-user  
- Less prone to cold-start issues  
- Can handle sparse data better  
- Offers **better RMSE and user coverage**  
- Works well when many users have rated few products  

### Evaluation:

- RMSE validated  
- **Top 20 products** generated per user  
- Applied sentiment analysis on those products  
- Final output: **Top 5 products** with highest % Positive Sentiment

---

## Recommendation System Validation

We **validated our recommendation system** on the **Top 1000 most active users**:

- For each user:
  - Generated Top 20 recommendations  
  - Filtered Top 5 using sentiment  
  - Calculated aggregate % Positive and % Negative      

### Validation Results:

**~85-90%** of recommendations were **highly positive** (> 85% positive sentiment)  
Recommendations were validated **against user history**  
Cold-start users handled well with item-item filtering  

---

## Web App — Flask Deployment

**Our system includes an interactive web application**:

- Built with **Flask**  
- Responsive Bootstrap UI  
- Features:
  - Autocomplete username input  
  - Submit button  
  - Display Top 5 recommendations  
  - Sidebar with already-rated products  

### Deployment:

Local deployment successful  
Heroku deployment live:

**Demo**: [https://product-recommendation-130dbb6b216e.herokuapp.com/](https://product-recommendation-130dbb6b216e.herokuapp.com/)

---

## Final Files & Structure

- `notebook.ipynb` — Full project notebook (EDA → modeling → evaluation → validation)  
- `model.py` — Final **sentiment model** and **recommendation system** code  
- `app.py` — Flask backend  
- `index.html` — Frontend (Bootstrap + jQuery UI)  
- `pickle/` — Trained model files (TF-IDF, models, rating matrices)  

---

## Conclusion

In this project, we successfully built a **complete Sentiment-Based Product Recommendation System**:

 Cleaned and processed raw review data  
 Trained and evaluated **multiple sentiment models**  
 Implemented **Item-Item Collaborative Filtering**  
 Validated recommendations on **Top 1000 users**  
 Built an interactive **Flask web app**  
 Deployed the app both **locally** and to **Heroku**

**Our system delivers accurate, personalized, sentiment-aware product recommendations** — helping Ebuss customers discover products they are likely to love.

---

**Thank you!**  
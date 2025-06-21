from flask import Flask, render_template, request
import pickle as pk
from model import recommend_products, get_user_rated_products

app = Flask(__name__)

# Load the UBCF matrix
final_rating = pk.load(open('pickle/item_final_rating.pkl', 'rb'))

@app.route('/')
def home():
    usernames = final_rating.index.tolist()
    print(usernames)
    return render_template('index.html', usernames=usernames)


# API endpoint for recommendations
@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    username = request.args.get('username')
    if username not in final_rating.index:
        return {'error': f"User '{username}' not found in recommendation system."}, 404
    try:
        recommendations_df = recommend_products(username)
    except Exception as e:
        return {'error': str(e)}, 500

    recommendations = recommendations_df.to_dict(orient='records')
    print(recommendations)
    return {
        'username': username,
        'recommendations': recommendations
    }

@app.route('/api/user_reviews', methods=['GET'])
def user_reviews():
    username = request.args.get('username')
    if username not in final_rating.index:
        return {'error': f"User '{username}' not found in recommendation system."}, 404
    try:
        user_rated_df = get_user_rated_products(username)
    except Exception as e:
        return {'error': str(e)}, 500

    # Convert to list of dicts
    rated_products = user_rated_df.to_dict(orient='records')

    return {
        'username': username,
        'rated_products': rated_products
    }

@app.route('/api/usernames', methods=['GET'])
def api_usernames():
    try:
        usernames = final_rating.index.tolist()
        term = request.args.get('term', '').lower()
        filtered_usernames = [u for u in usernames if term in u.lower()]
        return {'usernames': filtered_usernames[:50]}
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify,render_template
import os
import pandas as pd
from datetime import datetime
from recommendation.recommender import YelpRecommendationSystem
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the recommendation system
data_folder = os.getenv('DATA_FOLDER', 'data')
model_path = os.getenv('MODEL_PATH', 'model/yelp_model.pkl')
recommender = YelpRecommendationSystem(data_folder=data_folder, model_path=model_path)

# Initialize system (Spark, data, features, model)
recommender.initialize_spark()
recommender.load_data()
X_train_df, Y_train_df, X_val_df, Y_val_df, _ = recommender.prepare_features()
recommender.load_model()

# Load CF artifacts for recommendation (if they exist)
cf_artifacts_dir = "model/recommender_business"
try:
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    recommender.item_topk = load_pickle(f"{cf_artifacts_dir}/item_topk.pkl")
    print("CF artifacts loaded successfully for recommendations")
    cf_available = True
except Exception as e:
    print(f"CF artifacts not found: {e}")
    print("Recommendation feature will be limited")
    cf_available = False



@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    data = request.get_json()
    user_id = data.get('user_id')
    business_id = data.get('business_id')
    
    if not user_id or not business_id:
        return jsonify({'error': 'Missing user_id or business_id'}), 400
    
    try:
        result = recommender.predict_for_user_business(user_id, business_id)
        
        # Get user and business names
        user_name = recommender.user_name_map.get(user_id, 'N/A')
        business_name = recommender.business_name_map.get(business_id, 'N/A')
        actual = recommender.get_true_rating_from_val(user_id, business_id)
        
        return jsonify({
            'user_id': user_id,
            'user_name': user_name,
            'business_id': business_id,
            'business_name': business_name,
            'predicted_rating': float(result['predicted_rating']),
            'actual_rating': None if actual is None else float(actual),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

'''
def api_recommend():
    data = request.get_json()
    user_id = data.get('user_id')
    k = int(data.get('k', 5))
    
    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400
    
    if not cf_available:
        return jsonify({'error': 'Recommendation feature not available'}), 503
    
    try:
        recommendations = recommender.recommend_for_user(user_id, k)
        
        if not recommendations or not recommendations.get('recommendations'):
            return jsonify({'error': f'No recommendations found for user {user_id}'}), 404
        
        return jsonify({
            **recommendations,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500'''
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for recommendations"""
    data = request.get_json()
    user_id = data.get('user_id')
    k = int(data.get('k', 10))

    if not user_id:
        return jsonify({'error': 'Missing user_id'}), 400

    if not cf_available:
        return jsonify({'error': 'Recommendation feature not available'}), 503

    try:
        recs = recommender.recommend_for_user(user_id, k)

        if not recs or not recs.get('recommendations'):
            return jsonify({'error': f'No recommendations found for user {user_id}'}), 404

        # Add city & state
        city_map = getattr(recommender, 'business_city_map', {}) or {}
        state_map = getattr(recommender, 'business_state_map', {}) or {}
        recs_with_location = []
        for rec in recs['recommendations']:
            bid = rec.get('business_id')
            city = city_map.get(bid, 'N/A')
            state = state_map.get(bid, 'N/A')
            recs_with_location.append({**rec, 'city': city, 'state': state})

        return jsonify({
            'user_id': recs.get('user_id', user_id),
            'user_name': recs.get('user_name', recommender.user_name_map.get(user_id, 'N/A')),
            'k': k,
            'recommendations': recs_with_location,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommender.model is not None,
        'cf_available': cf_available,
        'timestamp': datetime.now().isoformat()
    })


@app.get("/")
def home():
    #return render_template("index.html", users=[], products=[])
    val_csv = os.path.join(recommender.data_folder, "yelp_val.csv")
    try:
        try:
            df_val = pd.read_csv(val_csv)  
            if not {"user_id","business_id"} <= set(df_val.columns):
                raise ValueError("missing headers")
        except Exception:
            df_val = pd.read_csv(val_csv, header=None, names=["user_id","business_id","stars"])

        df_first100 = df_val.head(100)


        val_autofill_map = {}
        for _, row in df_first100.iterrows():
            uid, bid = row["user_id"], row["business_id"]
            if uid not in val_autofill_map:
                val_autofill_map[uid] = {"bid": bid}
    except Exception as e:
        print("WARN building val_autofill_map:", e)
        val_autofill_map = {}


    users = list(val_autofill_map.keys())
    products = list({info["bid"] for info in val_autofill_map.values()})
    demo_pairs = [
    {
        "user_id": "DeXKbQYNx52OlOizobOLJw",
        "user_name": "Meghan",
        "business_id": "mnI_n7A8sxgOSmtgI3wzQQ",
        "business_name": "Amazing Cafe",
        "actual_rating": 5.0,
        "badges": ["Cafe", "Close match"]
    },
    {
        "user_id": "a6edJQpI-MEg-OUTKDp3HA",
        "user_name": "Allen",
        "business_id": "Ur6qSirXUZQgRMumvXJvrA",
        "business_name": "Lovelady Brewing Company",
        "actual_rating": 4.0,
        "badges": ["Brewing"]
    },
    {
        "user_id": "f_5VRh79aew1cVWUmC1PJA",
        "user_name": "Leah",
        "business_id": "Lcn8bGwxg3kilmtRbOn9ZQ",
        "business_name": "Southern Hills Animal Hospital",
        "actual_rating": 5.0,
        "badges": ["High confidence", "Animal hospital"]
    },
    {
        "user_id": "SVcQ8yqaLDmjMwPRDwQPaA",
        "user_name": "Chrisene-Faye",
        "business_id": "aYTiyhUbc6uL7jSAXGKhrA",
        "business_name": "Gelato Messina",
        "actual_rating": 5.0,
        "badges": ["Gelato"]
    },
    {
        "user_id": "lCmKYOdM5JV2UUmkVMwlKw",
        "user_name": "Caitlin",
        "business_id": "IVnGPHdTyu_GbLo9mXj98w",
        "business_name": "Ramen Tatsu",
        "actual_rating": 5.0,
        "badges": ["Ramen", "Close match"]
    },
    {
        "user_id": "ic-tyi1jElL_umxZVh8KNA",
        "user_name": "Owen",
        "business_id": "VGqzcHo_IJgRT0UV4oI52g",
        "business_name": "The P&L Burger",
        "actual_rating": 3.0,
        "badges": ["Burger", "High confidence"]
    },
    {
        "user_id": "Q1MSqlemsO43eKL_gjiAFA",
        "user_name": "Tien",
        "business_id": "N6MzDoao6s68nAIW8vS9dQ",
        "business_name": "Biaggio's Pizzeria",
        "actual_rating": 5.0,
        "badges": ["Pizzeria", "High confidence"]
    },
    {
        "user_id": "voNl6rXo9c-NuYIBIj5AQg",
        "user_name": "Bianca",
        "business_id": "SIWwh4m6kkvUaa5skrVAQQ",
        "business_name": "Jerry's Famous Coffee Shop",
        "actual_rating": 5.0,
        "badges": ["Coffee", "Close match"]
    },
    {
        "user_id": "zZYHZwmBl9Af4pI-aLXBFA",
        "user_name": "Jimmy",
        "business_id": "WAmCIDi3qn2VI7OUqZfjYQ",
        "business_name": "Solstice Tavern",
        "actual_rating": 3.0,
        "badges": ["Tavern", "Close match"]
    },
    {
        "user_id": "C2C0GPKvzWWnP57Os9eQ0w",
        "user_name": "Clint",
        "business_id": "mG7w2Ro7kOnrYUYlVTYtvA",
        "business_name": "KISS By Monster Mini Golf",
        "actual_rating": 4.0,
        "badges": ["Mini golf", "High confidence"]
    },
    {
        "user_id": "NS3B17yHv2lBJfGuwSmVNw",
        "user_name": "Cindy",
        "business_id": "QNgJwDus5DGSLjzaxotl9A",
        "business_name": "Heaven's Best Carpet Cleaning Las Vegas",
        "actual_rating": 5.0,
        "badges": ["Carpet cleaning", "Close match"]
    },
    {
        "user_id": "keBv05MsMFBd0Hu98vXThQ",
        "user_name": "Jesse",
        "business_id": "NCFwm2-TDb-oBQ2medmYDg",
        "business_name": "Fountains of Bellagio",
        "actual_rating": 5.0,
        "badges": ["High confidence", "Attraction"]
    },
    {
        "user_id": "j55ySt8CuDQ0bahoVBAb_A",
        "user_name": "Alice",
        "business_id": "Cy1bPenka65T2AUBAn-WkQ",
        "business_name": "T-Swirl Crêpe",
        "actual_rating": 3.0,
        "badges": ["High confidence", "Crêpe/Dessert"]
    },
    {
        "user_id": "CxDOIDnH8gp9KXzpBHJYXw",
        "user_name": "Jennifer",
        "business_id": "CuR4Xxu_aHrYD0IGGRdm5Q",
        "business_name": "South of Temperance",
        "actual_rating": 3.0,
        "badges": ["High confidence", "Bar/Restaurant"]
    },
    {
        "user_id": "GnbkuE6mWWBHr2UQR2KMnA",
        "user_name": "Chase",
        "business_id": "xfWdUmrz2ha3rcigyITV0g",
        "business_name": "Gordon Ramsay Burger",
        "actual_rating": 4.0,
        "badges": ["High confidence", "Burger"]
    },
    {
        "user_id": "MHn01HpwdCeaN3NbOVp0HA",
        "user_name": "Kady",
        "business_id": "Al-cFxIKoJpbPW9Roy0FEw",
        "business_name": "Swiss Chalet Rotisserie & Grill",
        "actual_rating": 2.0,
        "badges": ["High confidence", "Rotisserie"]
    },
    {
        "user_id": "kjaUSiRWhR9bF9KxOMbVvg",
        "user_name": "Michael",
        "business_id": "osu1j_Lg8R9brpuMMn3a3A",
        "business_name": "Fogo de Chão Brazilian Steakhouse",
        "actual_rating": None,
        "badges": ["Steakhouse", "Breakfast & Lunch"]
    },
    {
        "user_id": "kjaUSiRWhR9bF9KxOMbVvg",
        "user_name": "Michael",
        "business_id": "DbEszO3wk1xVmN3pCPob2g",
        "business_name": "Jamms Restaurant",
        "actual_rating": None,
        "badges": ["Steakhouse", "Breakfast & Lunch"]
    }
]


    return render_template(
        "index.html",
        users=users,
        products=products,
        val_autofill_map=val_autofill_map,
        demo_pairs=demo_pairs
    )


@app.post("/predict")
def predict_form():
    # pick manual value if typed, otherwise dropdown
    user_id = request.form.get("user_id_manual") or request.form.get("user_id")
    business_id = request.form.get("product_id_manual") or request.form.get("product_id")

    if not user_id or not business_id:
        return render_template(
            "result.html",
            user_id=user_id or "N/A",
            user_name="N/A",
            business_id=business_id or "N/A",
            business_name="N/A",
            predicted_rating="N/A",
            timestamp=datetime.now().isoformat(),
            error="Missing user_id or business_id"
        )

    try:
        result = recommender.predict_for_user_business(user_id, business_id)
        user_name = recommender.user_name_map.get(user_id, "N/A")
        business_name = recommender.business_name_map.get(business_id, "N/A")
        actual = recommender.get_true_rating_from_val(user_id, business_id)

        return render_template(
            "result.html",
            user_id=user_id,
            user_name=user_name,
            business_id=business_id,
            business_name=business_name,
            predicted_rating=round(float(result["predicted_rating"]), 2),
            actual_rating=("N/A" if actual is None else round(float(actual), 2)),
            timestamp=datetime.now().isoformat(),
            error=None
        )
    except Exception as e:
        return render_template(
            "result.html",
            user_id=user_id,
            user_name="N/A",
            business_id=business_id,
            business_name="N/A",
            predicted_rating="N/A",
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
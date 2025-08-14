# Yelp Recommendation System

A machine learning-based recommendation system for predicting Yelp user ratings of businesses.  
This project is containerized with Docker for consistent deployment across environments.

## Features

- Personalized rating predictions using collaborative filtering and business/user features  
- Flask + Gunicorn backend API  
- Apache Spark for scalable data processing  
- Dockerized environment for easy deployment  
- Preloaded trained model and dataset for out-of-the-box usage  
- Health check endpoint for service monitoring  

---

## Project Structure

```text
.
├── app.py                        # Flask application entry point
├── recommender.py                # Core recommendation logic
├── Dockerfile                    # Docker build instructions
├── docker-compose.yml            # Compose config for deployment
├── requirements.txt              # Python dependencies
├── .dockerignore                 # Docker ignore rules
├── readme.md                     # Project documentation
│
├── data/                         # Data files
│   ├── business.json             # Business information
│   ├── checkin.json              # Check-in data
│   ├── photo.json                # Photo metadata
│   ├── pure_jaccard_similarity.csv # Similarity metrics
│   ├── review_train.json         # Training reviews
│   ├── tip.json                  # Tips data
│   ├── user.json                 # User information
│   ├── yelp_train.csv            # Training dataset
│   ├── yelp_val.csv              # Validation dataset
│   └── yelp_val_in.csv           # Input validation dataset
│
├── model/                        # Trained models & artifacts
│   ├── yelp_model.pkl
│   └── recommender_business/...
│
├── output/                       # Output files
│   ├── low_error_demo.csv
│   ├── optimized_predictions.csv
│   ├── response_times.png
│   ├── top_predicted_10.csv
│   ├── top5_businesses.csv
│   └── top5_users.csv
│
├── templates/                    # HTML templates for Flask
│   ├── index.html                # Homepage search form
│   └── result.html               # Prediction result page
│
├── logs/                         # Runtime logs
│   ├── api.log
│   ├── recommendation_system.log
│   └── error_log.json
```

Local Development (Without Docker)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```


The app will be available at:
http://localhost:5000


## Docker Hub Image

The prebuilt image for this project is available on Docker Hub:  
[https://hub.docker.com/r/yuqizhang56/yelp-rec-spark](https://hub.docker.com/r/yuqizhang56/yelp-rec-spark)

## Pull and Run (Recommended for Users)

```bash
docker pull yuqizhang56/yelp-rec-spark:1.1
docker run -p 5000:5000 yuqizhang56/yelp-rec-spark:1.1
```
This will start the API on http://localhost:5000 without building the image locally.



# Deploy to Docker Hub (For Developers)

If you modify the code and want to push your own image:

docker login
docker tag yelp-rec-spark:1.1 yuqizhang56/yelp-rec-spark:1.1
docker push yuqizhang56/yelp-rec-spark:1.1



## API Endpoints
Health Check
curl http://localhost:5000/health

Predict Rating for a Specific User-Business Pair
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "48vRThjhuhiSQINQ2KV8Sw",
    "business_id": "fThrN4tfupIGetkrz18JOg"
  }'
```


Example Response:

```json
{
  "business_id": "fThrN4tfupIGetkrz18JOg",
  "business_name": "BRIM Kitchen + Brewery",
  "predicted_rating": 3.896714210510254,
  "timestamp": "2025-08-10T16:55:49.414171",
  "user_id": "48vRThjhuhiSQINQ2KV8Sw",
  "user_name": "Susan"
}
```

Recommend Top-K Businesses for a User
```bash
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "48vRThjhuhiSQINQ2KV8Sw",
    "k": 5
  }'
```


Example Response:

```json
{
  "recommendations": [
    {
      "business_id": "O7UMzd3i-Zk8dMeyY9ZwoA",
      "business_name": "Art of Flavors",
      "city": "Las Vegas",
      "predicted_rating": 4.818873405456543,
      "state": "NV"
    },
    {
      "business_id": "IT_4EEIbv6Ox1jBRMyE7pg",
      "business_name": "Del Frisco's Double Eagle Steak House",
      "city": "Las Vegas",
      "predicted_rating": 4.79454231262207,
      "state": "NV"
    },
    {
      "business_id": "IhNASEZ3XnBHmuuVnWdIwA",
      "business_name": "Brew Tea Bar",
      "city": "Las Vegas",
      "predicted_rating": 4.785144329071045,
      "state": "NV"
    },
    {
      "business_id": "A-uZAD4zP3rRxb44WUGV5w",
      "business_name": "Soho Japanese Restaurant",
      "city": "Las Vegas",
      "predicted_rating": 4.785010814666748,
      "state": "NV"
    },
    {
      "business_id": "igHYkXZMLAc9UdV5VnR_AA",
      "business_name": "Echo & Rig",
      "city": "Las Vegas",
      "predicted_rating": 4.719509124755859,
      "state": "NV"
    }
  ],
  "timestamp": "2025-08-10T16:56:35.484448",
  "user_id": "48vRThjhuhiSQINQ2KV8Sw",
  "user_name": "Susan"
}
```

Model Performance

RMSE: ~0.96 (validation)

Average API latency: ~34ms

Throughput: ~29 req/sec on local machine

## License

[MIT License](LICENSE)

## Contact

Yuqi Zhang - [yuqizhang56@outlook.com](mailto:yuqizhang56@outlook.com)


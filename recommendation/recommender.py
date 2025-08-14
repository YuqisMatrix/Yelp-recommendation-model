import sys
import time
import json
import numpy as np
import pandas as pd
from pyspark import SparkContext
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import math
import ast
import logging
import pickle
import os
import collections
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/recommendation_system.log')
    ]
)

logger = logging.getLogger('recommendation_system')



class YelpRecommendationSystem:
    def __init__(self, data_folder='data', model_path='model/yelp_model.pkl'):
        """
        Initialize the recommendation system with data paths

        Args:
            data_folder: Path to the folder containing data files
            model_path: Path to save/load the trained model
        """
        self.data_folder = data_folder
        self.model_path = model_path
        self.sc = None
        self.model = None
        self.feature_columns = None
        self.categorical_columns = [
            'credit_card', 'is_open', 'validated', 'noise_level',
            'delivery', 'takeout', 'wifi', 'table_service', 'wheelchair_accessible'
        ]
        self.user_name_map = {}
        self.business_name_map = {}
        self.user_business_dict = {}
        self.business_user_dict = {}
        self.rating_map = {}
        self.val_rating_map = {}


        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    

    def initialize_spark(self, app_name='YelpRecommendationSystem'):
        """Initialize Spark context with optimized configuration"""
        if self.sc is None:
            self.sc = SparkContext('local[*]', app_name)
            # Set memory options for better performance
            self.sc.setLogLevel("ERROR")  # Reduce log verbosity

        return self.sc

    def load_data(self, train_file=None, validation_file=None, business_file=None,
                  user_file=None, review_file=None, photo_file=None):
        """
        Load and preprocess data from files

        Args:
            train_file: Path to training data CSV
            validation_file: Path to validation data CSV
            business_file: Path to business data JSON
            user_file: Path to user data JSON
            review_file: Path to review data JSON
            photo_file: Path to photo data JSON

        Returns:
            Preprocessed data for model training
        """
        start_time = time.time()
        logger.info("Loading data...")

        # Set default file paths if not provided
        train_file = train_file or f"{self.data_folder}/yelp_train.csv"
        validation_file = validation_file or f"{self.data_folder}/yelp_val.csv"
        business_file = business_file or f"{self.data_folder}/business.json"
        user_file = user_file or f"{self.data_folder}/user.json"
        photo_file = photo_file or f"{self.data_folder}/photo.json"

        sc = self.initialize_spark()

        # Process data sequentially instead of in parallel to avoid Spark serialization issues

        # Load photo data first
        logger.info("Loading photo data...")
        self.bus_photo_numMap = self._load_photo_data(photo_file)

        # Load business data next (depends on photo data)
        logger.info("Loading business data...")
        self.bus_dict = self._load_business_data(business_file)

        # Load user data
        logger.info("Loading user data...")
        self.user_dict = self._load_user_data(user_file)

        # Load training data
        logger.info("Loading training data...")
        self.train_rdd = self._load_train_data(train_file)

        # Load validation data
        logger.info("Loading validation data...")
        self.validation_RDD = self._load_validation_data(validation_file)

        load_time = time.time() - start_time
        logger.info(f"Data loading completed in {load_time:.2f} seconds")

        return self



    def _load_photo_data(self, photo_file):
        """Load and process photo data"""
        photo_numRDD = self.sc.textFile(photo_file)
        photo_numRDD = photo_numRDD.map(lambda row: (json.loads(row)['business_id'], 1))
        return photo_numRDD.reduceByKey(lambda a, b: a + b).collectAsMap()

    def _load_business_data(self, business_file):
        """Load and process business data"""
        '''bus_RDD = self.sc.textFile(business_file)
        self.business_name_map = bus_RDD.map(
    lambda row: (json.loads(row)['business_id'], json.loads(row).get('name', ''))
    ).collectAsMap()'''
        bus_RDD = self.sc.textFile(business_file)

        # get (business_id, (name, city, state)) from JSON data
        bus_data = bus_RDD.map(lambda row: json.loads(row)) \
                      .map(lambda r: (
                          r['business_id'],
                          (r.get('name', ''), r.get('city', 'N/A'), r.get('state', 'N/A'))
                      )) \
                      .collectAsMap()

        # get business_id to name, city, state maps
        self.business_name_map = {bid: vals[0] for bid, vals in bus_data.items()}
        self.business_city_map = {bid: vals[1] for bid, vals in bus_data.items()}
        self.business_state_map = {bid: vals[2] for bid, vals in bus_data.items()}
        
        # Create a final copy of photo map to avoid serialization of self
        photo_map = self.bus_photo_numMap

        # Define parse function that doesn't reference self
        def parse_business_row(row):
            """Parse business JSON data into features"""
            row_dict = json.loads(row)
            business_id = row_dict['business_id']
            stars = float(row_dict['stars'])
            review_count = float(row_dict['review_count'])
            attributes = row_dict.get('attributes', {})
            photo_count = photo_map.get(business_id, 0)
            '''bus_json_RDD = self.sc.textFile(business_file).map(json.loads)
            self.business_city_map = bus_json_RDD.map(
            lambda r: (r['business_id'], r.get('city', 'N/A'))
            ).collectAsMap()

            self.business_state_map = bus_json_RDD.map(
            lambda r: (r['business_id'], r.get('state', 'N/A'))
            ).collectAsMap()'''
            

            # Calculate operation days
            time_dict = row_dict.get('hours', {})
            if time_dict is None:
                operation_days = 7
            elif not isinstance(time_dict, dict):
                operation_days = 0
            else:
                operation_days = len(time_dict.keys())
                if operation_days == 0:
                    operation_days = 7

            if attributes is None:
                price_range = None
                credit_card = None
                noise_level = None
                has_delivery = None
                has_takeout = None
                has_wifi = None
                validated = None
                table_service = None
                wheelchair_accessible = None
            else:
                price_range = attributes.get('RestaurantsPriceRange2', None)
                credit_card = attributes.get('BusinessAcceptsCreditCards', None)
                noise_level = attributes.get('NoiseLevel', None)
                has_delivery = attributes.get('RestaurantsDelivery', None)
                has_takeout = attributes.get('RestaurantsTakeOut', None)
                has_wifi = attributes.get('HasTV', None)
                business_parking = attributes.get('BusinessParking', None)
                table_service = attributes.get('RestaurantsTableService', None)
                wheelchair_accessible = attributes.get('WheelchairAccessible', None)

                if business_parking:
                    try:
                        business_parking = ast.literal_eval(business_parking)
                        validated = business_parking.get('validated', None)
                    except (ValueError, SyntaxError):
                        validated = None
                else:
                    validated = None

            is_open = row_dict['is_open']

            return (business_id, (stars, review_count, photo_count, operation_days,
                                  price_range, credit_card, is_open, validated,
                                  noise_level, has_delivery, has_takeout, has_wifi,
                                  table_service, wheelchair_accessible))

        # Apply the parse function and collect as map
        bus_RDD = bus_RDD.map(parse_business_row)
        return bus_RDD.collectAsMap()

    def _load_user_data(self, user_file):
        """Load and process user data"""
        user_RDD = self.sc.textFile(user_file)
        parsed_users = user_RDD.map(json.loads)
        self.user_name_map = parsed_users.map(lambda user: (
        user['user_id'],
        user.get('name', 'N/A')
        )).collectAsMap()
        '''user_RDD = self.sc.textFile(user_file) \
            .map(json.loads) \
            .map(lambda user: (
            user['user_id'],
            (
                float(user.get('average_stars', 3.0)),  # Default to 3.0 if missing
                float(user.get('review_count', 0)),
                float(user.get('useful', 0)),
                float(user.get('funny', 0)),  # Added feature
                float(user.get('cool', 0))  # Added feature
            )
        ))'''
        user_RDD = parsed_users.map(lambda user: (user['user_id'],
        (
        float(user.get('average_stars', 3.0)),
        float(user.get('review_count', 0)),
        float(user.get('useful', 0)),
        float(user.get('funny', 0)),
        float(user.get('cool', 0))
        )
))

        return user_RDD.collectAsMap()
    def _load_train_data(self, train_file):
        """Load training data"""
        train_rawRDD = self.sc.textFile(train_file)
        header = train_rawRDD.first()
        train_rdd = train_rawRDD.filter(lambda line: line != header).map(
        lambda x: (x.split(',')[0].strip(), x.split(',')[1].strip(), x.split(',')[2].strip())).cache()
        
        self.user_business_dict = train_rdd \
        .map(lambda x: (x[0], x[1])) \
        .groupByKey().mapValues(list) \
        .collectAsMap()

        self.business_user_dict = train_rdd \
        .map(lambda x: (x[1], x[0])) \
        .groupByKey().mapValues(list) \
        .collectAsMap()

        self.rating_map = train_rdd.map(
        lambda x: ((x[0], x[1]), float(x[2]))
            ).collectAsMap()
        # 1. business_user_rating_dict: business_id -> {user_id: rating}
        self.business_user_rating_dict = train_rdd.map(
        lambda x: (x[1], (x[0], float(x[2])))
        ).groupByKey().mapValues(lambda x: {user: rating for user, rating in x}).collectAsMap()
    
    # 2. business_avg_rating: business_id -> avg_rating  
        self.business_avg_rating = train_rdd.map(
        lambda x: (x[1], (float(x[2]), 1))
        ).reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
        .mapValues(lambda x: x[0]/x[1]).collectAsMap()
    
    # 3. user_avg_dict: user_id -> avg_rating
        self.user_avg_dict = train_rdd.map(
            lambda x: (x[0], (float(x[2]), 1))
        ).reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])) \
         .mapValues(lambda x: x[0]/x[1]).collectAsMap()

    # 4. global_avg
        total_rating_sum = train_rdd.map(lambda x: float(x[2])).reduce(lambda a, b: a + b)
        total_rating_count = train_rdd.count()
        self.global_avg = total_rating_sum / total_rating_count if total_rating_count > 0 else 0
        return train_rdd.collect()
    def get_user_businesses(self, user_id):
        business_ids = self.user_business_dict.get(user_id, [])
        return [
            {
                "business_id": bid,
                "business_name": self.business_name_map.get(bid, "N/A")
            }
            for bid in business_ids
        ]

    def get_business_users(self, business_id):
        user_ids = self.business_user_dict.get(business_id, [])
        return [
            {
                "user_id": uid,
                "user_name": self.user_name_map.get(uid, "N/A")
            }
            for uid in user_ids
        ]
    


    def _load_validation_data(self, validation_file):
        """Load validation data"""
        raw_val_RDD = self.sc.textFile(validation_file)
        header = raw_val_RDD.first()
        validation_RDD = raw_val_RDD.filter(lambda row: row != header).map(
            lambda row: row.split(",")
        ).collect()
        try:
            df_val = pd.read_csv(validation_file)
            self.val_rating_map = {
                (row['user_id'], row['business_id']): float(row['stars'])
                for _, row in df_val.iterrows()
            }
        except Exception as e:
            logger.warning(f"Failed to build val_rating_map: {e}")
            self.val_rating_map = {}

        return validation_RDD
    def get_true_rating_from_val(self, user_id, business_id):
        return self.val_rating_map.get((user_id, business_id))


    def prepare_features(self):
        """Prepare feature datasets for training and validation"""
        logger.info("Preparing features for training and validation...")
        start_time = time.time()

        # Prepare training features
        X_train_data_list, Y_train_data_list = self._prepare_training_features()

        # Define feature columns
        base_columns = [
            'user_avg_star', 'user_review_cnt', 'user_useful', 'user_funny', 'user_cool',
            'bus_stars', 'review_count_bus', 'photo_count', 'operation_days',
            'price_range', 'credit_card', 'is_open', 'validated', 'noise_level',
            'delivery', 'takeout', 'wifi', 'table_service', 'wheelchair_accessible'
        ]

        # Create training dataframes
        X_train_df = pd.DataFrame(X_train_data_list, columns=base_columns)

        # Add interaction features
        X_train_df['user_bus_rating_diff'] = X_train_df['user_avg_star'] - X_train_df['bus_stars']
        X_train_df['user_bus_rating_ratio'] = X_train_df['user_avg_star'] / (X_train_df['bus_stars'] + 0.1)
        X_train_df['user_activity_ratio'] = X_train_df['user_review_cnt'] / (X_train_df['user_useful'] + 1)

        # Convert categorical features
        X_train_df = pd.get_dummies(
            X_train_df,
            columns=self.categorical_columns,
            drop_first=True
        )

        # Store feature columns for later prediction
        self.feature_columns = X_train_df.columns

        # Convert target variable
        Y_train_df = pd.DataFrame(Y_train_data_list, columns=['rating'])
        Y_train_df['rating'] = pd.to_numeric(Y_train_df['rating'], errors='coerce')

        # Prepare validation features
        validation_list, X_val_list = self._prepare_validation_features()
        X_val_df = pd.DataFrame(X_val_list, columns=base_columns)

        # Add same interaction features to validation data
        X_val_df['user_bus_rating_diff'] = X_val_df['user_avg_star'] - X_val_df['bus_stars']
        X_val_df['user_bus_rating_ratio'] = X_val_df['user_avg_star'] / (X_val_df['bus_stars'] + 0.1)
        X_val_df['user_activity_ratio'] = X_val_df['user_review_cnt'] / (X_val_df['user_useful'] + 1)

        # Ensure validation data has same dummy columns as training
        X_val_df = pd.get_dummies(X_val_df, columns=self.categorical_columns, drop_first=True)

        # Align validation columns with training columns
        for col in self.feature_columns:
            if col not in X_val_df.columns:
                X_val_df[col] = 0
        X_val_df = X_val_df[self.feature_columns]

        # Create Y validation dataframe from validation file
        df_val = pd.read_csv(f"{self.data_folder}/yelp_val.csv")
        true_ratings = df_val['stars'].tolist()
        Y_val_df = pd.DataFrame(true_ratings, columns=['rating'])
        Y_val_df['rating'] = pd.to_numeric(Y_val_df['rating'], errors='coerce')

        logger.info(f"Feature preparation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Training data shape: {X_train_df.shape}")
        logger.info(f"Validation data shape: {X_val_df.shape}")

        return X_train_df, Y_train_df, X_val_df, Y_val_df, validation_list

    def _prepare_training_features(self):
        """Prepare features for training data"""
        X_train_data_list = []
        Y_train_data_list = []

        for user, bus, rating_train in self.train_rdd:
            Y_train_data_list.append(rating_train)

            # Get user information
            if user in self.user_dict:
                user_average, user_rvw_num, user_useful, user_funny, user_cool = self.user_dict[user]
            else:
                user_average = 3.0  # Default to average rating
                user_rvw_num = user_useful = user_funny = user_cool = 0

            # Get business information
            if bus in self.bus_dict:
                bus_info = self.bus_dict[bus]
                (bus_stars, rvw_num_bus, photo_count, operation_days, price_range,
                 credit_card, is_open, validated, noise_level, delivery,
                 takeout, wifi, table_service, wheelchair_accessible) = bus_info

                # Process credit card feature
                credit_card = int(credit_card == "True" or credit_card == True)

                # Process price range feature
                if isinstance(price_range, str):
                    try:
                        price_range = int(price_range)
                    except ValueError:
                        price_range = 2  # Default to medium price range
                elif price_range is None:
                    price_range = 2  # Default value
            else:
                # Default values if business not found
                bus_stars = 3.5  # Default to average business rating
                rvw_num_bus = photo_count = 0
                operation_days = 7  # Default to full week
                price_range = 2  # Default to medium price range
                credit_card = is_open = 1  # Default to yes
                validated = noise_level = delivery = takeout = wifi = table_service = wheelchair_accessible = None

            # Append data to X_train_data
            X_train_data_list.append([
                user_average, user_rvw_num, user_useful, user_funny, user_cool,
                bus_stars, rvw_num_bus, photo_count, operation_days, price_range,
                credit_card, is_open, validated, noise_level, delivery,
                takeout, wifi, table_service, wheelchair_accessible
            ])

        return X_train_data_list, Y_train_data_list

    def _prepare_validation_features(self):
        """Prepare features for validation data"""
        validation_list = []
        X_val_list = []

        for lines in self.validation_RDD:
            user = lines[0]
            bus = lines[1]
            validation_list.append((user, bus))

            # Get user information
            if user in self.user_dict:
                user_average, user_rvw_num, user_useful, user_funny, user_cool = self.user_dict[user]
            else:
                user_average = 3.0  # Default to average rating
                user_rvw_num = user_useful = user_funny = user_cool = 0

            # Get business information
            if bus in self.bus_dict:
                bus_info = self.bus_dict[bus]
                (bus_stars, rvw_num_bus, photo_count, operation_days, price_range,
                 credit_card, is_open, validated, noise_level, delivery,
                 takeout, wifi, table_service, wheelchair_accessible) = bus_info

                # Process credit card feature
                credit_card = int(credit_card == "True" or credit_card == True)

                # Process price range feature
                if isinstance(price_range, str):
                    try:
                        price_range = int(price_range)
                    except ValueError:
                        price_range = 2  # Default to medium price range
                elif price_range is None:
                    price_range = 2  # Default value
            else:
                # Default values if business not found
                bus_stars = 3.5  # Default to average business rating
                rvw_num_bus = photo_count = 0
                operation_days = 7  # Default to full week
                price_range = 2  # Default to medium price range
                credit_card = is_open = 1  # Default to yes
                validated = noise_level = delivery = takeout = wifi = table_service = wheelchair_accessible = None

            # Append data to X_val_list
            X_val_list.append([
                user_average, user_rvw_num, user_useful, user_funny, user_cool,
                bus_stars, rvw_num_bus, photo_count, operation_days, price_range,
                credit_card, is_open, validated, noise_level, delivery,
                takeout, wifi, table_service, wheelchair_accessible
            ])

        return validation_list, X_val_list

    def train_model(self, X_train_df, Y_train_df, X_val_df, Y_val_df):
        """Train the XGBoost model with optimized parameters"""
        logger.info("Training model...")
        start_time = time.time()

        # Optimized parameters based on experimentation
        xgb_params = {
            'objective': 'reg:squarederror',  # Updated objective function
            'random_state': 2024,
            'min_child_weight': 5,
            'n_estimators': 300,  # Reduced for faster runtime
            'learning_rate': 0.08,  # Increased learning rate
            'max_depth': 7,  # Reduced depth to prevent overfitting
            'gamma': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.3,
            'reg_lambda': 1.2,
            'scale_pos_weight': 1,
            'tree_method': 'hist',  # Faster algorithm
            'early_stopping_rounds': 50  # Early stopping to prevent overfitting
        }
        
        self.model = XGBRegressor(**xgb_params)

        # Train with evaluation set for early stopping
        self.model.fit(
            X_train_df, Y_train_df,
            eval_set=[(X_val_df, Y_val_df)],
            verbose=False
        )

        logger.info(f"Model training completed in {time.time() - start_time:.2f} seconds")

        # Save the model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {self.model_path}")

        return self.model

    def load_model(self):
        """Load a pre-trained model"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded from {self.model_path}")
            return True
        else:
            logger.warning(f"Model file {self.model_path} not found")
            return False

    def predict(self, X_val_df, validation_list=None, output_path=None):
        """Make predictions and optionally save results"""
        logger.info("Making predictions...")
        start_time = time.time()

        # Ensure model is loaded
        if self.model is None:
            loaded = self.load_model()
            if not loaded:
                logger.error("No model available for prediction")
                return None

        # Make predictions
        Y_pred = self.model.predict(X_val_df)
        Y_pred = np.clip(Y_pred, None, 5.0)

        # Save predictions if output path is provided
        if output_path and validation_list:
            result_str = "user_id,business_id,prediction\n"
            for pred, (user_id, business_id) in zip(Y_pred, validation_list):
                result_str += f"{user_id},{business_id},{pred}\n"

            with open(output_path, "w") as f:
                f.write(result_str)
            logger.info(f"Predictions saved to {output_path}")

        logger.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
        return Y_pred

    def evaluate(self, predictions, ground_truth):
        """Evaluate model performance"""
        logger.info("Evaluating model performance...")

        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)

        # Calculate RMSE
        mse = mean_squared_error(ground_truth, predictions)
        rmse = math.sqrt(mse)

        # Calculate error distribution
        abs_diff = np.abs(predictions - ground_truth)
        error_levels = {
            '0-1': 0,
            '1-2': 0,
            '2-3': 0,
            '3-4': 0,
            '4+': 0
        }

        # Count errors by level
        for error in abs_diff:
            if 0 <= error < 1:
                error_levels['0-1'] += 1
            elif 1 <= error < 2:
                error_levels['1-2'] += 1
            elif 2 <= error < 3:
                error_levels['2-3'] += 1
            elif 3 <= error < 4:
                error_levels['3-4'] += 1
            elif error >= 4:
                error_levels['4+'] += 1

        # Log high-error cases
        high_error_indices = np.where(abs_diff >= 3)[0]
        for i in high_error_indices:
            logger.warning(
                f"High Error Case: Predicted: {predictions[i]:.2f}, Actual: {ground_truth[i]:.2f}, Error: {abs_diff[i]:.2f}")

        # Log results
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info("Error Distribution:")
        for level, count in error_levels.items():
            logger.info(f"{level}: {count} ({count / len(predictions) * 100:.2f}%)")

        return rmse, error_levels
    def calculate_similarity(self, test_business, other_business, co_rated_users):
        """Calculate similarity between two businesses based on co-rated users"""
        avg_i = self.business_avg_rating[test_business]
        avg_j = self.business_avg_rating[other_business]
    
        sum_xy = sum_x2 = sum_y2 = 0.0
        for u in co_rated_users:
            x = self.business_user_rating_dict[test_business][u] - avg_i
            y = self.business_user_rating_dict[other_business][u] - avg_j
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y
    
        denom = math.sqrt(sum_x2) * math.sqrt(sum_y2)
        if denom == 0:
            return None
    
        w = sum_xy / denom
        return w * w  # squared correlation
    def precompute_item_similarities(self, topk=50, min_cousers=20):
        """Precompute top-k similar items for each business"""
        logger.info("Precomputing item similarities...")
        start_time = time.time()
    
        self.item_topk = {}
        all_items = list(self.business_user_dict.keys())
    
        for i, b in enumerate(all_items):
            if i % 100 == 0:  # Log progress every 100 items
                logger.info(f"Processing business {i}/{len(all_items)}")
            
            sims = []
            users_b = set(self.business_user_dict[b])
        
            for other in all_items:
                if other == b:
                    continue
                co = users_b.intersection(set(self.business_user_dict[other]))
                if len(co) < min_cousers:
                    continue
                s = self.calculate_similarity(b, other, co)
                if s is None:
                    continue
                sims.append((other, s))
        
            sims.sort(key=lambda t: -t[1])
            self.item_topk[b] = sims[:topk]
    
        logger.info(f"Item similarities precomputed in {time.time() - start_time:.2f} seconds")
    
    def save_cf_artifacts(self, art_dir="model/recommender_business"):
        """Save CF dictionaries and precomputed similarities"""
        import os
        import pickle
    
        os.makedirs(art_dir, exist_ok=True)
    
        def save_pickle(obj, path):
            with open(path, "wb") as f:
                pickle.dump(obj, f)
    
        logger.info("Saving CF artifacts...")
        # Save CF dictionaries
        save_pickle(self.user_business_dict, f"{art_dir}/user_business_dict.pkl")
        save_pickle(self.business_user_dict, f"{art_dir}/business_user_dict.pkl")
        save_pickle(self.business_user_rating_dict, f"{art_dir}/business_user_rating_dict.pkl")
        save_pickle(self.business_avg_rating, f"{art_dir}/business_avg_rating.pkl")
        save_pickle(self.user_avg_dict, f"{art_dir}/user_avg_dict.pkl")
        save_pickle(self.global_avg, f"{art_dir}/global_avg.pkl")
        save_pickle(self.business_name_map, f"{art_dir}/bus_name_dict.pkl")
        save_pickle(self.user_name_map, f"{art_dir}/user_name_dict.pkl")
        save_pickle(self.item_topk, f"{art_dir}/item_topk.pkl")
        save_pickle(self.business_city_map, f"{art_dir}/business_city_map.pkl")
        save_pickle(self.business_state_map, f"{art_dir}/business_state_map.pkl")
    
        logger.info(f"✅ CF artifacts saved to {art_dir}")
    def get_candidates_for_user(self, user_id, per_item_k=20, max_pool=300):
        """Generate candidate businesses for recommendation based on user's rating history"""
        # get user rated businesses
        rated_businesses = self.user_business_dict.get(user_id, [])
        if not rated_businesses:
        # if user has no rated businesses, return top businesses based on average rating
            top_items = sorted(self.business_avg_rating.items(), key=lambda t: -t[1])[:max_pool]
            return [bid for bid, _ in top_items]
    
    # based on user's rated businesses, get similar businesses
        candidates = set()
    
        for business in rated_businesses:
        # get top similar businesses for this business
            similar_businesses = self.item_topk.get(business, [])
            for similar_bid, similarity in similar_businesses[:per_item_k]:
                if similar_bid not in rated_businesses:  # exclude already rated businesses
                    candidates.add(similar_bid)
            for similar_bid, similarity in similar_businesses[:per_item_k]:
                if similar_bid not in rated_businesses:  # exclude already rated businesses
                    candidates.add(similar_bid)

                if len(candidates) >= max_pool:
                    break
        
            if len(candidates) >= max_pool:
                break
    
        return list(candidates)
    
    def recommend_for_user(self, user_id, k=10):
        """Recommend top-k businesses for a user with predicted ratings"""
        logger.info(f"Generating recommendations for user: {user_id}")
    
    # 1. Get candidate businesses for the user
        candidates = self.get_candidates_for_user(user_id)
        logger.info(f"Generated {len(candidates)} candidates for user {user_id}")
    
        if not candidates:
            return []

    # 2. Use XGBoost model to predict ratings for each candidate business
        scored_candidates = []
        failed_predictions = 0
    
        for business_id in candidates:
            try:
                prediction_result = self.predict_for_user_business(user_id, business_id)
                if prediction_result:
                    scored_candidates.append({
                        'business_id': business_id,
                        'business_name': self.business_name_map.get(business_id, 'N/A'),
                        'predicted_rating': float(prediction_result['predicted_rating'])
                    })
                else:
                    failed_predictions += 1
            except Exception as e:
                logger.warning(f"Error predicting for {user_id}, {business_id}: {e}")
                failed_predictions += 1
                continue

        logger.info(f"Successfully predicted {len(scored_candidates)} businesses, {failed_predictions} failed")

    # 3. Sort by predicted rating and return top-k
        scored_candidates.sort(key=lambda x: -x['predicted_rating'])
    
        return {
            'user_id': user_id,
            'user_name': self.user_name_map.get(user_id, 'N/A'),
            'recommendations': scored_candidates[:k]
    }
    
    
    def run_pipeline(self, train=True, output_path='predictions.csv'):
        """Run the full recommendation system pipeline"""
        total_start_time = time.time()

        # 1) Load data and prepare features
        self.load_data()
        X_train_df, Y_train_df, X_val_df, Y_val_df, validation_list = self.prepare_features()

        # 2) Train the model or load an existing one
        if train:
            self.train_model(X_train_df, Y_train_df, X_val_df, Y_val_df)
        else:
            self.load_model()

        # 3) Generate predictions
        predictions = self.predict(X_val_df, validation_list, output_path)

        # 4) Read the true ratings from the validation file
        df_val = pd.read_csv(f"{self.data_folder}/yelp_val.csv")
        ground_truth = df_val['stars'].tolist()

        # 5) Build a results DataFrame
        df_results = pd.DataFrame({
            'user_id':     [u for u, b in validation_list],
            'business_id': [b for u, b in validation_list],
            'pred_rating': predictions,
            'true_rating': ground_truth
        })
        # Add user names and business names
        df_results['user_name']     = df_results['user_id'].map(self.user_name_map)
        df_results['business_name'] = df_results['business_id'].map(self.business_name_map)

    # 6) Export the Top-10 “High Confidence” examples
        df_top10 = df_results.sort_values('pred_rating', ascending=False).head(10)
        top10_fp = os.path.join(os.path.dirname(output_path), 'top_predicted_10.csv')
        df_top10.to_csv(
            top10_fp,
            index=False,
            columns=[
                'user_id', 'business_id',
                'pred_rating', 'true_rating',
                'user_name', 'business_name'
            ]
        )
        logger.info(f"Top-Predicted 10 exported to {top10_fp}")
        # After mapping user_name & business_name:
        df_results['error'] = (df_results['pred_rating'] - df_results['true_rating']).abs()

        intervals = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (4.9, 5.1)]
        selected_frames = []
        for low, high in intervals:
            mask = (df_results['pred_rating'] >= low) & (df_results['pred_rating'] < high) & (df_results['error'] < 0.1)
            top3 = df_results[mask].nsmallest(3, 'error').copy()
            top3['interval'] = f"{low}-{high}"
            selected_frames.append(top3)
        df_low_error = pd.concat(selected_frames, ignore_index=True)


        low_error_fp = os.path.join(os.path.dirname(output_path), 'low_error_demo.csv')
        df_low_error.to_csv(low_error_fp, index=False)
        logger.info(f"Low-Error samples exported to {low_error_fp}")


        # 7) Evaluate model performance and return metrics
        rmse, error_levels = self.evaluate(predictions, ground_truth)
        # 8) Precompute item similarities for recommendation
        self.precompute_item_similarities()

        total_runtime = time.time() - total_start_time
        logger.info(f"Total pipeline runtime: {total_runtime:.2f} seconds")
        # 9) Save CF artifacts
        self.save_cf_artifacts()

        return {
            'rmse': rmse,
            'error_distribution': error_levels,
            'runtime': total_runtime
        }
    def predict_for_user_business(self, user_id, business_id):
        """Make a single prediction for a given user and business"""
        # Ensure model and features are loaded
        if self.model is None or self.feature_columns is None:
            logger.info("Features not initialized, preparing features...")
            X_train_df, Y_train_df, X_val_df, Y_val_df, _ = self.prepare_features()

            if self.model is None:
                loaded = self.load_model()
                if not loaded:
                    logger.error("No model available for prediction")
                    return None

        # Get user information
        if user_id in self.user_dict:
            user_average, user_rvw_num, user_useful, user_funny, user_cool = self.user_dict[user_id]
        else:
            user_average = 3.0  # Default values
            user_rvw_num = user_useful = user_funny = user_cool = 0

        # Get business information
        if business_id in self.bus_dict:
            bus_info = self.bus_dict[business_id]
            (bus_stars, rvw_num_bus, photo_count, operation_days, price_range,
             credit_card, is_open, validated, noise_level, delivery,
             takeout, wifi, table_service, wheelchair_accessible) = bus_info

            # Process credit card feature
            credit_card = int(credit_card == "True" or credit_card == True)

            # Process price range feature
            if isinstance(price_range, str):
                try:
                    price_range = int(price_range)
                except ValueError:
                    price_range = 2  # Default value
            elif price_range is None:
                price_range = 2  # Default value
        else:
            # Default values if business not found
            bus_stars = 3.5  # Default
            rvw_num_bus = photo_count = 0
            operation_days = 7  # Default
            price_range = 2  # Default
            credit_card = is_open = 1  # Default
            validated = noise_level = delivery = takeout = wifi = table_service = wheelchair_accessible = None

        # Create a single feature row
        features = {
            'user_avg_star': user_average,
            'user_review_cnt': user_rvw_num,
            'user_useful': user_useful,
            'user_funny': user_funny,
            'user_cool': user_cool,
            'bus_stars': bus_stars,
            'review_count_bus': rvw_num_bus,
            'photo_count': photo_count,
            'operation_days': operation_days,
            'price_range': price_range,
            'credit_card': credit_card,
            'is_open': is_open,
            'validated': validated,
            'noise_level': noise_level,
            'delivery': delivery,
            'takeout': takeout,
            'wifi': wifi,
            'table_service': table_service,
            'wheelchair_accessible': wheelchair_accessible
        }

        # Create DataFrame
        X_df = pd.DataFrame([features])

        # Add interaction features
        X_df['user_bus_rating_diff'] = X_df['user_avg_star'] - X_df['bus_stars']
        X_df['user_bus_rating_ratio'] = X_df['user_avg_star'] / (X_df['bus_stars'] + 0.1)
        X_df['user_activity_ratio'] = X_df['user_review_cnt'] / (X_df['user_useful'] + 1)

        # Convert categorical features
        X_df = pd.get_dummies(X_df, columns=self.categorical_columns, drop_first=True)

        # Align columns with training features
        for col in self.feature_columns:
            if col not in X_df.columns:
                X_df[col] = 0
        X_df = X_df[self.feature_columns]

        # Make prediction
        prediction = self.model.predict(X_df)[0]

        return {
            'user_id': user_id,
            'business_id': business_id,
            'predicted_rating': prediction
        }


    def predict_user_on_others_businesses(self, target_user_id, reference_user_id, include_truth=True):
        """Predict target_user_id's rating for every business the reference user rated."""
    # 1) businesses Carrie (reference) has rated
        ref_businesses = self.user_business_dict.get(reference_user_id, [])
        rows = []

        for bid in ref_businesses:
            # predicted rating for Lisa (target) on this business
            pred = self.predict_for_user_business(target_user_id, bid)['predicted_rating']

        # optional: grab true stars if they exist in train
            ref_true  = self.rating_map.get((reference_user_id, bid)) if include_truth else None
            targ_true = self.rating_map.get((target_user_id, bid))   if include_truth else None

            rows.append({
            'business_id':   bid,
            'business_name': self.business_name_map.get(bid, 'N/A'),
            'predicted_for_target': float(pred),
            'ref_user_true': ref_true,      # Carrie's actual rating (if any)
            'target_true':   targ_true      # Lisa's actual rating (if any)
            })

    # 2) sort by predicted rating desc
        rows.sort(key=lambda r: r['predicted_for_target'], reverse=True)
        return rows


# Main execution

if __name__ == "__main__":
    start_time = time.time()

    # get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # get the project root directory (one level up)
    project_root = os.path.dirname(script_dir)

    # Check command line arguments
    data_folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(project_root, 'data')
    validation_fp = sys.argv[2] if len(sys.argv) > 2 else os.path.join(data_folder, 'yelp_val.csv')

    # Modify output path to ensure it points to the output folder under the project root
    output_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(project_root, 'output', 'optimized_predictions.csv')

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Create and run recommendation system
    recommendation_system = YelpRecommendationSystem(data_folder=data_folder)
    results = recommendation_system.run_pipeline(train=True, output_path=output_path)

    # Print summary
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"RMSE: {results['rmse']:.6f}")
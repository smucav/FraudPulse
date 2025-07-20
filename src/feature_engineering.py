# src/feature_engineering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class FeatureEngineer:
    """Class to handle feature engineering for fraud detection."""
    
    @staticmethod
    def add_time_features(df, purchase_time_col, signup_time_col):
        """
        Add time-based features: hour_of_day, day_of_week, time_since_signup.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            purchase_time_col (str): Purchase timestamp column.
            signup_time_col (str): Signup timestamp column.
        
        Returns:
            pd.DataFrame: DataFrame with time-based features.
        """
        df['hour_of_day'] = df[purchase_time_col].dt.hour
        df['day_of_week'] = df[purchase_time_col].dt.day_name()
        df['time_since_signup'] = (df[purchase_time_col] - df[signup_time_col]).dt.total_seconds() / 3600
        print("Added time-based features: hour_of_day, day_of_week, time_since_signup")
        return df
    
    @staticmethod
    def add_behavioral_features(df, user_id_col, purchase_value_col, time_since_signup_col):
        """
        Add behavioral features: transaction_count, velocity.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            user_id_col (str): User ID column.
            purchase_value_col (str): Purchase value column.
            time_since_signup_col (str): Time since signup column.
        
        Returns:
            pd.DataFrame: DataFrame with behavioral features.
        """
        # Transaction frequency per user
        df['transaction_count'] = df.groupby(user_id_col)[user_id_col].transform('count')
        # Velocity: purchase value per hour since signup
        df['velocity'] = df[purchase_value_col] / (df[time_since_signup_col] + 1e-6)
        print("Added behavioral features: transaction_count, velocity")
        return df
    
    @staticmethod
    def encode_and_scale(df, categorical_cols, numerical_cols):
        """
        Encode categorical features and scale numerical features.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            categorical_cols (list): Categorical column names.
            numerical_cols (list): Numerical column names.
        
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        # Keep top 10 countries to reduce dimensionality
        top_countries = df['country'].value_counts().head(10).index.tolist()
        df['country'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')
        
        # One-hot encoding
        encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
        encoded_features = pd.DataFrame(
            encoder.fit_transform(df[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        df = df.reset_index(drop=True)
        encoded_features = encoded_features.reset_index(drop=True)
        df = df.drop(categorical_cols, axis=1)
        df = pd.concat([df, encoded_features], axis=1)
        
        # Scaling numerical features
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        print("Encoded categorical and scaled numerical features")
        return df

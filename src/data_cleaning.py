# src/data_cleaning.py
import pandas as pd
import numpy as np

class DataCleaner:
    """Class to handle data cleaning tasks for fraud detection datasets."""
    
    @staticmethod
    def drop_missing_and_duplicates(df):
        """
        Remove missing values and duplicates from the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        # Drop rows with missing values to ensure data integrity
        df_cleaned = df.dropna().copy()
        # Remove duplicates to prevent bias in fraud detection
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"Shape after cleaning: {df_cleaned.shape}")
        return df_cleaned
    
    @staticmethod
    def convert_timestamps(df, timestamp_cols):
        """
        Convert specified columns to datetime format.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            timestamp_cols (list): Columns to convert to datetime.
        
        Returns:
            pd.DataFrame: DataFrame with converted timestamps.
        """
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col])
        print(f"Converted {timestamp_cols} to datetime")
        return df
    
    @staticmethod
    def convert_ip_columns(df, ip_cols):
        """
        Convert IP address columns to integer.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            ip_cols (list): IP address column names.
        
        Returns:
            pd.DataFrame: DataFrame with converted IP columns.
        """
        for col in ip_cols:
            df[col] = df[col].astype(int)
        print(f"Converted {ip_cols} to integer")
        return df
    
    @staticmethod
    def map_ip_to_country(df, ip_col, ip_mapping):
        """
        Map IP addresses to countries using vectorized operation.
        
        Args:
            df (pd.DataFrame): Input DataFrame with IP addresses.
            ip_col (str): IP address column name.
            ip_mapping (pd.DataFrame): IP-to-country mapping DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with new 'country' column.
        """
        def find_country(ip):
            match = ip_mapping[(ip_mapping['lower_bound_ip_address'] <= ip) & 
                              (ip_mapping['upper_bound_ip_address'] >= ip)]
            return match.iloc[0]['country'] if not match.empty else 'Other'
        
        df['country'] = df[ip_col].apply(find_country)
        print(f"Unique countries mapped: {df['country'].nunique()}")
        return df

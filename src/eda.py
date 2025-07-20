# src/eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    """Class to perform exploratory data analysis for fraud detection."""
    
    @staticmethod
    def plot_class_distribution(df, class_col, title, save_path):
        """
        Plot class distribution for fraud detection.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            class_col (str): Class column name.
            title (str): Plot title.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(6, 4))
        sns.countplot(x=class_col, data=df)
        plt.title(title)
        plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
        plt.ylabel('Count')
        plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_bivariate_boxplot(df, x_col, y_col, title, save_path, log_scale=False):
        """
        Plot boxplot of a numerical feature vs. class.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            x_col (str): Class column name.
            y_col (str): Numerical column name.
            title (str): Plot title.
            save_path (str): Path to save the plot.
            log_scale (bool): Apply log scale to y-axis.
        """
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=x_col, y=y_col, data=df)
        if log_scale:
            plt.yscale('log')
        plt.title(title)
        plt.xlabel('Class (0: Non-Fraud, 1: Fraud)')
        plt.ylabel(y_col)
        plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_country_vs_class(df, class_col, title, save_path):
        """
        Plot fraud count by country (top 10).
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            class_col (str): Class column name.
            title (str): Plot title.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(8, 5))
        sns.countplot(y='country', hue=class_col, data=df, order=df['country'].value_counts().index[:10])
        plt.title(title)
        plt.xlabel('Count')
        plt.ylabel('Country')
        plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_correlation_heatmap(df, numerical_cols, title, save_path):
        """
        Plot correlation heatmap for numerical features.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            numerical_cols (list): Numerical column names.
            title (str): Plot title.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
        plt.title(title)
        plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def compute_class_imbalance(df, class_col):
        """
        Compute and print class imbalance ratio.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            class_col (str): Class column name.
        
        Returns:
            float: Fraud ratio.
        """
        counts = df[class_col].value_counts(normalize=True)
        ratio = counts[1]
        print(f"{class_col} Imbalance: {ratio:.4f} (Fraud) vs {1-ratio:.4f} (Non-Fraud)")
        return ratio

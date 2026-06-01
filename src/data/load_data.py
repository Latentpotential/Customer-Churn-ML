import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    
    return pd.read_csv(path)
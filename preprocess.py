import pandas as pd
import json
import re
import logging
import os
from datetime import datetime
from sys import exit

# Function to parse each line and extract needed data
def parse_line(line, columns_to_load):
    data = json.loads(line)
    return {col: data[col] for col in columns_to_load}

def clean_newlines(text):
    """ Clean newlines from text fields, replace them with a single space, and also collapse multiple spaces into one. """
    # Replace newlines with space and collapse multiple spaces into one
    cleaned_text = re.sub(r'\s+', ' ', text)
    
    # Remove leading spaces if present
    return cleaned_text.lstrip()

def get_latest_version_date(versions):
    """ Extract the date of the latest version from the versions list. """
    if versions:
        latest_version = max(versions, key=lambda x: datetime.strptime(x['created'], '%a, %d %b %Y %H:%M:%S GMT'))
        return datetime.strptime(latest_version['created'], '%a, %d %b %Y %H:%M:%S GMT').date()
    return None

def get_csv():
    # File path of the JSON file
    file_path = 'arxiv-metadata-oai-snapshot.json'

    if not os.path.exists(file_path):
        logging.error("Dataset not found. Please download the datset from https://www.kaggle.com/datasets/Cornell-University/arxiv "
                      "and extract the zip directly in the root folder.")
        exit()
    else:
        logging.info("The JSON already exists.")

    # Specify the columns to load
    columns_to_load = ['title', 'categories', 'abstract', 'versions']

    # Read the file line by line and parse JSON objects
    logging.debug("Read files from JSON.")
    with open(file_path, 'r') as file:
        data = [parse_line(line, columns_to_load) for line in file]

    # Convert the list of dictionaries into a DataFrame
    logging.debug("Creating DataFrame.")
    df = pd.DataFrame(data)

    # Clean text fields in the DataFrame
    logging.debug("Clean text.")
    df['title'] = df['title'].apply(clean_newlines)
    df['categories'] = df['categories'].apply(clean_newlines)
    df['abstract'] = df['abstract'].apply(clean_newlines)

    # Extract the date of the latest version and update the DataFrame
    logging.debug("Extract latest version and convert date format.")
    df['versions'] = df['versions'].apply(get_latest_version_date)

    # Save the DataFrame to a CSV file
    logging.debug("Save DataFrame as CSV.")
    df.to_csv('arxiv-metadata-oai-snapshot.csv', index=False)

def get_data():
    # File path of the CSV file
    file_path = 'arxiv-metadata-oai-snapshot.csv'
    # Check if the file exists
    if not os.path.exists(file_path):
        # If the file does not exist, call the get_csv method
        logging.info("Creating CSV from JSON file.")
        get_csv()
    else:
        logging.info("The CSV already exists.")

    # Read the CSV file into a DataFrame
    logging.info("Reading CSV.")
    df = pd.read_csv(file_path)

    # Display the first few rows of the DataFrame to confirm it's loaded correctly
    logging.info(str(df.head()))

    return df

def main():
    pass

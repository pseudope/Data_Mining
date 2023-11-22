import logging

from preprocess import get_data
import embedding
import interprete
import similarity
import visualize

LOG_LEVEL = logging.DEBUG

# Set up logging configuration
logging.basicConfig(level=LOG_LEVEL, format='%(levelname)s: %(message)s')

def main():
    df = get_data()

if __name__ == "__main__":
    main()
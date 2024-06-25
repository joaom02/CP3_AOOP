# Reddit Sentiment Analysis Tool

This sentiment analysis tool is designed to assess the emotional tone behind a series of comments posted on Reddit. It utilizes machine learning models and natural language processing techniques to determine if the sentiment of each comment is positive, neutral, or negative.

## Features

- Fetches comments from a specified Reddit thread URL.
- Analyzes each comment for sentiment.
- Displays the overall sentiment of all comments.
- Shows the title of the Reddit thread in the results.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.6+
- pip
- Virtual environment (recommended)

## Installation

To install the Reddit Sentiment Analysis Tool, follow these steps:

1. Clone the repository:

https://github.com/joaom02/CP3_AOOP.git

2. Navigate to the project directory:

cd ./src/

3. Install the required packages:

pip install -r requirements.txt


## Usage

To use the Reddit Sentiment Analysis Tool, follow these steps:

1. Start the Flask application:

python main.py

2. Open a web browser and navigate to `http://127.0.0.1:5000/`.
3. Enter the URL of the Reddit post you want to analyze in the form provided on the home page.
4. Submit the form to see the sentiment analysis results.

## How it Works

- The tool uses Selenium to scrape comments from the provided Reddit URL.
- It applies a pre-trained TensorFlow model to predict sentiment using a TF-IDF vectorized representation of each comment.
- The overall sentiment is calculated based on the most frequent sentiment among all analyzed comments.

## Contributing

Contributions to this project are welcome. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your_feature`).
3. Make changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your_feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

If you have any questions or feedback, please contact me at joao.m@ipvc.pt.

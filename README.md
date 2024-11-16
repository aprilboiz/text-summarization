# text-summarization

This simple extractive text summarization tool is implemented from scratch using Python, powered by TF-IDF.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/text-summarization.git
    cd text-summarization
    ```

2. **Create a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Setup spaCy

1. **Download the spaCy language model**:
    ```sh
    python -m spacy download en_core_web_sm
    ```

## Usage
There are 2 ways to run this application:
1. **Run the summarization script**:
    ```sh
    python summarizer.py [--top_n TOP_N] input_file
    ```

2. **Through the Flask server**:
This also provides an API to summarize a YouTube video using its subtitle (en-us only).
   ```sh
    flask --app main.py --debug run
    ```

# text-summarization

This is a simple text summarization tool implemented from scratch using Python, powered by TF-IDF.

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

1. **Run the summarization script**:
    ```sh
    python summarizer.py [--top_n TOP_N] input_file
    ```

2. **Through the Flask server**
   ```sh
    flask --app main.py --debug run
    ```

import spacy
import re
from collections import Counter
import json
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class TextSummarizer:
    def __init__(
        self, model_name: str = "en_core_web_sm", cache_dir: Optional[Path] = None
    ):
        self.nlp = spacy.load(model_name)
        self.cache_dir = cache_dir
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def load_text(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise FileNotFoundError(f"Error loading file {file_path}: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        # Convert to lowercase and strip whitespace
        text = text.strip()

        # Remove URLs
        text = re.sub(r"http\S+|www.\S+", "", text)

        # Replace newlines and multiple spaces
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep sentence-ending punctuation
        text = re.sub(r"[^\w\s.!?]", "", text)

        # Normalize sentence endings
        text = re.sub(r"([.!?])\s*([A-Za-z])", r"\1 \2", text)

        return text

    def clean_sentences(self, text: str) -> Tuple[List[str], List[str]]:
        """Improved sentence cleaning with better token filtering."""
        doc = self.nlp(text)

        # Original sentences for final output
        original_sentences = [sent.text.strip() for sent in doc.sents]

        # Clean and preprocess sentences
        cleaned_sentences = []
        for sent in doc.sents:
            cleaned_tokens = [
                token.lemma_.lower()
                for token in sent
                if (
                    not token.is_stop
                    and not token.is_punct
                    and not token.is_space
                    and len(token.text) > 1
                    and token.has_vector
                )
            ]
            if cleaned_tokens:
                cleaned_sentences.append(" ".join(cleaned_tokens))

        return cleaned_sentences, original_sentences

    def calculate_tf(self, sentences: List[str]) -> Dict[str, Dict[str, float]]:
        tf_matrix = {}

        for sent in sentences:
            doc = self.nlp(sent)

            # Count tokens
            token_counts = Counter(
                token.text.lower()
                for token in doc
                if not token.is_space and len(token.text) > 1
            )

            # Calculate frequency with smoothing
            total_tokens = sum(token_counts.values())
            tf_matrix[sent] = {
                token: (count + 1) / (total_tokens + len(token_counts))
                for token, count in token_counts.items()
            }

        return tf_matrix

    def calculate_idf(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate IDF with improved document frequency handling."""
        total_docs = len(sentences)
        word_doc_count = Counter()

        # Count document frequency for each word
        for sent in sentences:
            # Use set to count unique words per sentence
            unique_words = set(
                token.text.lower()
                for token in self.nlp(sent)
                if not token.is_space and len(token.text) > 1
            )
            word_doc_count.update(unique_words)

        # Calculate IDF with smoothing
        idf_matrix = {
            word: math.log((total_docs + 1) / (count + 1))
            for word, count in word_doc_count.items()
        }

        return idf_matrix

    def calculate_tf_idf(
        self, tf_matrix: Dict[str, Dict[str, float]], idf_matrix: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate TF-IDF with proper normalization."""
        tf_idf_matrix = {}

        for sent, tf_scores in tf_matrix.items():
            # Calculate TF-IDF scores
            tf_idf_scores = {
                token: tf_score * idf_matrix.get(token, 0)
                for token, tf_score in tf_scores.items()
            }

            # L2 normalization
            magnitude = math.sqrt(sum(score**2 for score in tf_idf_scores.values()))
            if magnitude > 0:
                tf_idf_scores = {
                    token: score / magnitude for token, score in tf_idf_scores.items()
                }

            tf_idf_matrix[sent] = tf_idf_scores

        return tf_idf_matrix

    def score_sentences(
        self,
        tf_idf_matrix: Dict[str, Dict[str, float]],
        sentences: Dict[str, str],
        top_n: int = 5,
    ) -> str:
        # Calculate sentence scores
        sentence_scores = []
        for cleaned_sent, tf_idf_scores in tf_idf_matrix.items():
            # Calculate weighted score
            score = sum(tf_idf_scores.values())
            position_boost = 1.0

            # Add position boost for first few sentences
            if cleaned_sent in list(sentences.keys())[:3]:
                position_boost = 1.25

            final_score = score * position_boost
            sentence_scores.append((cleaned_sent, final_score))

        # Dynamic threshold based on standard deviation
        scores = [score for _, score in sentence_scores]
        threshold = np.mean(scores) + 0.5 * np.std(scores)

        # Sort and select sentences
        selected_sents = []
        sentence_scores.sort(key=lambda x: x[1], reverse=True)

        for cleaned_sent, score in sentence_scores[:top_n]:
            if score > threshold:
                original_sent = sentences.get(cleaned_sent, "")
                if original_sent:
                    selected_sents.append(original_sent)

        # Ensure proper sentence ordering
        if selected_sents:
            # Sort selected sentences by their original position
            sent_positions = {sent: idx for idx, sent in enumerate(sentences.values())}
            selected_sents.sort(key=lambda x: sent_positions.get(x, float("inf")))

        return " ".join(selected_sents)

    def summarize(self, file_path: str, top_n: int = 5) -> str:
        """Main summarization method with error handling and caching."""
        try:
            # Load and process text
            text = self.load_text(file_path)
            text = self.preprocess_text(text)

            # Clean sentences
            cleaned_sents, original_sentences = self.clean_sentences(text)
            if not cleaned_sents:
                raise ValueError("No valid sentences found in text")

            # Create mapping of cleaned to original sentences
            sent_mapping = dict(zip(cleaned_sents, original_sentences))

            # Calculate matrices
            tf_matrix = self.calculate_tf(cleaned_sents)
            idf_matrix = self.calculate_idf(cleaned_sents)
            tf_idf_matrix = self.calculate_tf_idf(tf_matrix, idf_matrix)

            # Cache results if cache directory is set
            if self.cache_dir:
                self._cache_matrices(tf_matrix, idf_matrix, tf_idf_matrix)

            # Generate summary
            summary = self.score_sentences(tf_idf_matrix, sent_mapping, top_n)
            return summary

        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")

    def _cache_matrices(self, tf_matrix, idf_matrix, tf_idf_matrix):
        """Cache calculated matrices."""
        cache_files = {
            "tf.json": tf_matrix,
            "idf.json": idf_matrix,
            "tf_idf.json": tf_idf_matrix,
        }

        for filename, data in cache_files.items():
            cache_path = Path(self.cache_dir) / filename
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=4)


if __name__ == "__main__":
    summarizer = TextSummarizer(cache_dir="cache")
    try:
        summary = summarizer.summarize("text2.txt", top_n=5)
        print("Summary:", summary)
    except Exception as e:
        print(f"Error: {str(e)}")

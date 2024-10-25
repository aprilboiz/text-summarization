from flask import Flask, jsonify, request
from summarizer import TextSummarizer

app = Flask(__name__)


class SummarizeType:
    TEXT = "text"
    VIDEO = "video"


@app.route("/api/summarize", methods=["POST"])
def summarize():
    summarizer = TextSummarizer()
    try:
        data = request.get_json()
        type = data.get("type", SummarizeType.TEXT)
        if type == SummarizeType.VIDEO:
            return jsonify({"error": "Video summarization not supported yet"})

        if type == SummarizeType.TEXT:
            text = data["text"]
            top_n = data.get("top_n", 5)
            summary = summarizer.summarize(text, top_n)
            return jsonify({"summary": summary})

        return jsonify({"error": "Invalid summarization type"})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)

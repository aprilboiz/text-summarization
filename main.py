from flask import Flask, jsonify, request
from summarizer import TextSummarizer
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)
from pytubefix import YouTube
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


class SummarizeType:
    TEXT = "text"
    VIDEO = "video"


def extract_video_id(url: str):
    video_id = re.search(r"watch\?v=(\w+)", url)
    if video_id:
        return video_id.group(1)
    return None


def is_youtube_url(url: str):
    return (
        re.match(r"https?://(www\.)?(youtube\.com|youtu\.be)/watch\?v=", url)
        is not None
    )


def construct_youtube_url(video_id: str):
    return f"https://youtube.com/watch?v={video_id}"


@app.route("/api/summarize", methods=["POST"])
def summarize():
    summarizer = TextSummarizer()
    try:
        data = request.get_json()
        type = data.get("type", SummarizeType.TEXT)
        if type == SummarizeType.VIDEO:
            try:
                video_id = data["video_id"]
            except KeyError:
                return jsonify({"error": "Missing 'video_id' field in request"}), 400
            top_n = data.get("top_n", 5)

            # Check if the video ID is a YouTube URL
            if is_youtube_url(video_id):
                video_id = extract_video_id(video_id)
                if video_id is None:
                    return jsonify({"error": "Invalid YouTube video URL"}), 400

            # If not, assume it's a video ID, and construct the YouTube URL
            video_url = construct_youtube_url(video_id)
            video = YouTube(video_url)
            try:
                video.title
            except VideoUnavailable:
                return jsonify({"error": "Video is unavailable"}), 400

            # Get the transcript
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=["en"]
                )
                text = " ".join([t["text"] for t in transcript])
                summary = summarizer.summarize(text, top_n)
                return jsonify({"summary": summary})
            except TranscriptsDisabled:
                return (
                    jsonify({"error": "Transcripts are disabled for this video"}),
                    400,
                )
            except NoTranscriptFound:
                return jsonify({"error": "No transcript found for this video"}), 400

        if type == SummarizeType.TEXT:
            try:
                text = data["text"]
            except KeyError:
                return jsonify({"error": "Missing 'text' field in request"}), 400
            top_n = data.get("top_n", 5)
            summary = summarizer.summarize(text, top_n)
            return jsonify({"summary": summary})

        return jsonify({"error": "Invalid summarization type"})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)

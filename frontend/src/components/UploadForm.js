import { useState, useRef } from "react";
import API from "../services/api";

function UploadForm() {
    const [video, setVideo] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [dragActive, setDragActive] = useState(false);
    const inputRef = useRef(null);

    const handleUpload = async () => {
        if (!video) {
            setError("Please select a video file");
            return;
        }

        setLoading(true);
        setError("");
        const formData = new FormData();
        formData.append("file", video);

        try {
            const response = await API.post("/analyze-video", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            setResult(response.data);
        } catch (err) {
            console.error(err);
            setError("Analysis failed. Try another video.");
        } finally {
            setLoading(false);
        }
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setVideo(e.dataTransfer.files[0]);
        }
    };



    return (
        <div className="page-container">
          <div className="content-wrapper">
            <div className="left-column">
              <label 
                className={`custom-upload-card ${dragActive ? 'dragover' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  accept="video/*"
                  className="hidden-file-input"
                  ref={inputRef}
                  onChange={(e) => setVideo(e.target.files ? e.target.files[0] : null)}
                />
                {video ? (
                  <>
                    <div className="video-preview-wrapper">
                      <video 
                        src={URL.createObjectURL(video)} 
                        className="video-preview"
                        controls 
                        muted
                      />
                    </div>
                    <p className="file-name">{video.name}</p>
                  </>
                ) : (
                  <div className="upload-content">
                    <div className="upload-icons">
                      📱 🎥
                    </div>
                    <h3>Click or drag & drop video here</h3>
                    <p>MP4, MOV, AVI (max 100MB recommended)</p>
                  </div>
                )}
              </label>

              <div className="analyze-btn-container">
                <button 
                  className="analyze-btn" 
                  disabled={!video || loading}
                  onClick={handleUpload}
                >
                  {loading ? "Analyzing..." : "🚀 Analyze Video"}
                </button>
              </div>

            </div>

            <div className="right-column">
              {error && (
                <div className="error-card">
                  {error}
                </div>
              )}

              <div className="result-card transcript-card">
                <h3>📝 Transcript</h3>
                <div className="transcript-content">
                  {result?.transcript || "No transcript available yet"}
                </div>
              </div>
              <div className="result-card sentiment-card">
                <h3>😊 Sentiment</h3>
                <div className={`sentiment-value ${result?.sentiment ? result.sentiment.label.toUpperCase() : ''}`}>
                  {result?.sentiment ? `${result.sentiment.label.replace(/^\w/, c => c.toUpperCase())} ${Math.round(result.sentiment.score * 100)}%` : "No sentiment yet"}
                </div>
              </div>
              <div className="result-card emotion-card">
                <h3>🎭 Dominant Emotion</h3>
                <div className="emotion-value">
                  {result?.video_emotion ? result.video_emotion.replace(/^\w/, c => c.toUpperCase()) : "No emotion yet"}
                </div>
              </div>
            </div>
          </div>
        </div>
    );
}

export default UploadForm;


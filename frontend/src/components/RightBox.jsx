import React, { useState, useEffect } from "react";
import axios from "axios";

const RightBox = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [downloadLink, setDownloadLink] = useState("");
  const [progress, setProgress] = useState({ step: "idle", message: "Waiting for upload..." });

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  useEffect(() => {
    // Create an EventSource to listen for progress updates
    const eventSource = new EventSource("http://localhost:8000/progress");

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("Received progress update:", data); // Log received data
      setProgress(data);
    };

    eventSource.onerror = (error) => {
      console.error("EventSource failed:", error); // Log errors
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, []);

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    setUploading(true);
    setDownloadLink("");

    try {
      const response = await axios.post("http://localhost:8000/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Response:", response); // Debugging log

      if (response.data.download_url) {
        setDownloadLink(response.data.download_url);
      } else {
        alert("No download link received from server!");
      }
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Failed to upload video.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="container right-box-animation">
      <div className="header">
        <input type="file" accept="video/*" onChange={handleFileChange} />
      </div>
      <button className="upload-btn" onClick={handleUpload} disabled={uploading}>
        {uploading ? "Uploading..." : "Upload Video"}
      </button>
      {downloadLink && (
        <a href={downloadLink} className="download-link" download>
          Download Processed Video
        </a>
      )}
      <div className="progress-status">
        <strong>Status:</strong> {progress.message}
      </div>
    </div>
  );
};

export default RightBox;
import React from "react";
import previewImage from "../images/img1.jpg";
import './leftbox.css'

function LeftBox() {
  return (
    <div className="card">
      <img src={previewImage} alt="OpusAI Preview" className="rounded-lg w-full h-auto mb-3" />

      <p className="heading">What is OpusAI?</p>
      <p>OpusAI enhances your videos by generating emotion-aware soundtracks.</p>
      <p>Upload your video and experience AI-driven audio adaptation!</p>
    </div>
  );
}

export default LeftBox;

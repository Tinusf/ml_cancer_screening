import React, { useState } from 'react';
import './App.css';
import Webcam from "react-webcam";

const App = props => {
  let results = "testy";

  const changeResults = props.changeResults;

  async function query(data) {
    const response = await fetch("https://skinflask.tinusf.com/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json"
      },
      body: JSON.stringify({
        data
      })
    });
    results = await response.json();
    changeResults(results);
    console.log(results);
  }

  const webcamRef = React.useRef(null);
  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current.getScreenshot();
    query(imageSrc);
  },
    [webcamRef]
  );
  return (
    <div>
      <Webcam
        className="webcam"
        audio={false}
        width={1280}
        height={720}
        ref={webcamRef}
        screenshotFormat="image/png"
        forceScreenshotSourceSize
        videoConstraints={{
          width: 28,
          height: 28,
        }}
      />
      <p>
        Check for skin cancer!
        </p>
      <button className="large green button" onClick={capture}>Capture</button>
    </div>
  );
}

export default App;

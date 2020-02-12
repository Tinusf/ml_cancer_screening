import React, { useState } from 'react';
import logo from './logo.svg';
import './App.css';
import Webcam from "react-webcam";


const App = props => {
  console.log(props)
  console.log("ræv")
  let results = "testy";

  const changeResults = props.changeResults;

  async function query(data) {
    const response = await fetch("http://localhost:8080", {
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
      <button onClick={capture}>Capture</button>
    </div>
  );
}

export default App;

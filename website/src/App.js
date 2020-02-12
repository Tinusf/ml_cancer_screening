import React from 'react';
import logo from './logo.svg';
import './App.css';
import Webcam from "react-webcam";

function App() {
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
    const json = await response.json();
    console.log(json);
    return json ? json.data : json;
  }

  const webcamRef = React.useRef(null);
  const capture = React.useCallback(
    () => {
      const imageSrc = webcamRef.current.getScreenshot();
      const res = query(imageSrc);
      console.log(res)
      console.log(imageSrc);
    },
    [webcamRef]
  );
  return (
    <div className="App">
      <header className="App-header">
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
      </header>
    </div>
  );
}

export default App;

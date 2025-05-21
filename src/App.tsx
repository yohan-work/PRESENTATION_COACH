import React, { useState, useEffect, useRef } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";

// Define SpeechRecognition type
interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  onresult:
    | ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any)
    | null;
  start(): void;
  stop(): void;
}

interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
  readonly isFinal: boolean;
}

interface SpeechRecognitionAlternative {
  readonly transcript: string;
  readonly confidence: number;
}

// Extend window object
declare global {
  interface Window {
    SpeechRecognition: { new (): SpeechRecognition };
    webkitSpeechRecognition: { new (): SpeechRecognition };
  }
}

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [presentationText, setPresentationText] = useState("");
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(
    null
  );
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const [transcript, setTranscript] = useState("");
  const [recognition, setRecognition] = useState<SpeechRecognition | null>(
    null
  );
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [cameraPermissionError, setCameraPermissionError] = useState<
    string | null
  >(null);

  // Handle audio permissions
  useEffect(() => {
    const setupAudio = async () => {
      if (!navigator.mediaDevices) {
        setPermissionError("Media devices not supported in your browser");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: true,
        });

        // Determine the appropriate MIME type
        const mimeTypes = ["audio/webm", "audio/mp4", "audio/ogg", "audio/wav"];
        let mimeType = "";

        for (const type of mimeTypes) {
          if (MediaRecorder.isTypeSupported(type)) {
            mimeType = type;
            break;
          }
        }

        if (!mimeType) {
          console.warn("No suitable MIME type found for MediaRecorder");
        }

        const recorderOptions = mimeType ? { mimeType } : undefined;
        const recorder = new MediaRecorder(stream, recorderOptions);

        console.log(
          `MediaRecorder initialized with MIME type: ${recorder.mimeType}`
        );

        // Move event listeners to the useEffect for recording state
        setMediaRecorder(recorder);
        setPermissionError(null);
      } catch (error) {
        console.error("Error accessing media devices.", error);
        if (error instanceof DOMException && error.name === "NotAllowedError") {
          setPermissionError(
            "Microphone access denied. Please allow microphone access in your browser settings to record audio."
          );
        } else {
          setPermissionError(
            `Error accessing microphone: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
        }
      }
    };

    setupAudio();
  }, []);

  // Handle video/camera permissions separately
  useEffect(() => {
    const setupCamera = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraPermissionError("Camera not supported in your browser");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setCameraPermissionError(null);
      } catch (error) {
        console.error("Error accessing camera:", error);
        if (error instanceof DOMException && error.name === "NotAllowedError") {
          setCameraPermissionError(
            "Camera access denied. Please allow camera access in your browser settings."
          );
        } else {
          setCameraPermissionError(
            `Error accessing camera: ${
              error instanceof Error ? error.message : String(error)
            }`
          );
        }
      }
    };

    setupCamera();
  }, []);

  useEffect(() => {
    const SpeechRecognitionConstructor =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognitionConstructor) {
      const recognitionInstance = new SpeechRecognitionConstructor();
      recognitionInstance.continuous = true;
      recognitionInstance.interimResults = true;
      recognitionInstance.onresult = (event: SpeechRecognitionEvent) => {
        const currentTranscript = Array.from(event.results)
          .map((result) => result[0].transcript)
          .join("");
        setTranscript(currentTranscript);
      };
      setRecognition(recognitionInstance);
    } else {
      console.error("Speech Recognition API not supported in this browser.");
    }
  }, []);

  useEffect(() => {
    const setupCamera = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      }
    };

    const loadModel = async () => {
      const model = await faceLandmarksDetection.createDetector(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
        {
          runtime: "tfjs",
          refineLandmarks: true,
        }
      );
      return model;
    };

    setupCamera();
    loadModel();
  }, []);

  useEffect(() => {
    const setBackend = async () => {
      await tf.setBackend("webgl");
      await tf.ready();
    };
    setBackend();
  }, []);

  // Add back the recording state handler effect
  useEffect(() => {
    if (isRecording && mediaRecorder) {
      setAudioChunks([]);

      // Add event listeners before starting
      const handleDataAvailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          setAudioChunks((chunks) => [...chunks, event.data]);
        }
      };

      mediaRecorder.addEventListener("dataavailable", handleDataAvailable);

      // Start recording with a small time slice to get frequent data
      mediaRecorder.start(200);

      console.log("Recording started");

      return () => {
        mediaRecorder.removeEventListener("dataavailable", handleDataAvailable);
      };
    } else if (
      !isRecording &&
      mediaRecorder &&
      mediaRecorder.state === "recording"
    ) {
      console.log("Stopping recording");
      mediaRecorder.stop();
    }
  }, [isRecording, mediaRecorder]);

  // Add a dedicated effect to handle the creation of the audio blob when recording stops
  useEffect(() => {
    if (!isRecording && audioChunks.length > 0) {
      console.log(`Creating audio blob from ${audioChunks.length} chunks`);

      // Add a small delay to ensure all chunks are collected
      const timer = setTimeout(() => {
        if (audioChunks.length > 0) {
          const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
          console.log(`Audio blob created: ${audioBlob.size} bytes`);

          const url = URL.createObjectURL(audioBlob);
          setAudioUrl(url);
          setShowResults(true);
        }
      }, 300);

      return () => clearTimeout(timer);
    }
  }, [isRecording, audioChunks]);

  const handleStart = () => {
    if (permissionError) {
      alert("Cannot start recording: " + permissionError);
      return;
    }

    setIsRecording(true);
    setShowResults(false);
    if (recognition) {
      try {
        recognition.start();
      } catch (error) {
        console.error("Error starting speech recognition:", error);
      }
    }
  };

  const handleStop = () => {
    setIsRecording(false);
    if (recognition) {
      try {
        recognition.stop();
      } catch (error) {
        console.error("Error stopping speech recognition:", error);
      }
    }
    console.log("Final Transcript:", transcript);
  };

  const handleReset = () => {
    setShowResults(false);
    setTranscript("");
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
  };

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPresentationText(event.target.value);
  };

  // Function to request permissions again
  const requestPermissions = async () => {
    setPermissionError(null);
    setCameraPermissionError(null);

    // Retry audio setup
    try {
      const audioStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      // Determine the appropriate MIME type
      const mimeTypes = ["audio/webm", "audio/mp4", "audio/ogg", "audio/wav"];
      let mimeType = "";

      for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          mimeType = type;
          break;
        }
      }

      const recorderOptions = mimeType ? { mimeType } : undefined;
      const recorder = new MediaRecorder(audioStream, recorderOptions);

      // We'll set event listeners in the recording useEffect
      setMediaRecorder(recorder);
    } catch (error) {
      setPermissionError(
        `Microphone access denied: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }

    // Retry camera setup
    try {
      const videoStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = videoStream;
      }
    } catch (error) {
      setCameraPermissionError(
        `Camera access denied: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Presentation Practice Mode</h1>

        {/* Show permission errors if any */}
        {(permissionError || cameraPermissionError) && (
          <div className="permission-errors">
            <h2>Permission Issues</h2>
            {permissionError && <p className="error">{permissionError}</p>}
            {cameraPermissionError && (
              <p className="error">{cameraPermissionError}</p>
            )}
            <button onClick={requestPermissions}>Grant Permissions</button>
            <p>
              Note: You'll need to enable microphone and camera permissions in
              your browser. Click the camera/microphone icon in your browser's
              address bar and allow access.
            </p>
          </div>
        )}

        {!showResults ? (
          <>
            <textarea
              value={presentationText}
              onChange={handleTextChange}
              placeholder="Enter your presentation text here..."
              rows={10}
              cols={50}
            />
            <div>
              <button
                onClick={handleStart}
                disabled={isRecording || !!permissionError}
              >
                Start
              </button>
              <button onClick={handleStop} disabled={!isRecording}>
                Stop
              </button>
            </div>
            <p>Transcript: {transcript}</p>
            {!cameraPermissionError ? (
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                width="640"
                height="480"
              />
            ) : (
              <div className="video-placeholder">
                <p>Camera unavailable. {cameraPermissionError}</p>
              </div>
            )}
          </>
        ) : (
          <div className="results-view">
            <h2>Presentation Results</h2>
            <div>
              <h3>Your Recording</h3>
              {audioUrl ? (
                <audio src={audioUrl} controls />
              ) : (
                <p>
                  No audio recording available.{" "}
                  {permissionError || "Recording failed."}
                </p>
              )}
            </div>
            <div>
              <h3>Your Transcript</h3>
              {transcript ? (
                <p>{transcript}</p>
              ) : (
                <p>
                  No transcript available. Speech recognition may not be
                  supported or enabled.
                </p>
              )}
            </div>
            <div>
              <h3>Feedback</h3>
              {transcript ? (
                <>
                  <p>
                    Speech clarity:{" "}
                    {transcript.length > 100 ? "Good" : "Needs improvement"}
                  </p>
                  <p>
                    Speaking pace:{" "}
                    {transcript.split(" ").length /
                      Math.max(1, audioChunks.length * 0.1)}{" "}
                    words per second
                  </p>
                </>
              ) : (
                <p>Feedback unavailable without transcript data.</p>
              )}
            </div>
            <button onClick={handleReset}>Practice Again</button>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;

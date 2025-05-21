import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

// Define SpeechRecognition type
interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
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
  const [presentationText, setPresentationText] = useState('');
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  const [transcript, setTranscript] = useState('');
  const [recognition, setRecognition] = useState<SpeechRecognition | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    if (isRecording && mediaRecorder) {
      mediaRecorder.start();
      mediaRecorder.ondataavailable = (event) => {
        setAudioChunks((prev) => [...prev, event.data]);
      };
    } else if (mediaRecorder) {
      mediaRecorder.stop();
    }
  }, [isRecording, mediaRecorder]);

  useEffect(() => {
    if (!navigator.mediaDevices) {
      console.error('Media devices not supported');
      return;
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
      .then((stream) => {
        const recorder = new MediaRecorder(stream);
        setMediaRecorder(recorder);
      })
      .catch((error) => console.error('Error accessing media devices.', error));
  }, []);

  useEffect(() => {
    const SpeechRecognitionConstructor = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognitionConstructor) {
      const recognitionInstance = new SpeechRecognitionConstructor();
      recognitionInstance.continuous = true;
      recognitionInstance.interimResults = true;
      recognitionInstance.onresult = (event: SpeechRecognitionEvent) => {
        const currentTranscript = Array.from(event.results)
          .map((result) => result[0].transcript)
          .join('');
        setTranscript(currentTranscript);
      };
      setRecognition(recognitionInstance);
    } else {
      console.error('Speech Recognition API not supported in this browser.');
    }
  }, []);

  useEffect(() => {
    const setupCamera = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      }
    };

    const loadModel = async () => {
      const model = await faceLandmarksDetection.createDetector(faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, {
        runtime: 'tfjs',
        refineLandmarks: true,
      });
      return model;
    };

    setupCamera();
    loadModel();
  }, []);

  useEffect(() => {
    const setBackend = async () => {
      await tf.setBackend('webgl');
      await tf.ready();
    };
    setBackend();
  }, []);

  const handleStart = () => {
    setIsRecording(true);
    if (recognition) {
      recognition.start();
    }
  };

  const handleStop = () => {
    setIsRecording(false);
    if (mediaRecorder) {
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
      };
    }
    if (recognition) {
      recognition.stop();
    }
    console.log('Final Transcript:', transcript);
  };

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPresentationText(event.target.value);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Presentation Practice Mode</h1>
        <textarea
          value={presentationText}
          onChange={handleTextChange}
          placeholder="Enter your presentation text here..."
          rows={10}
          cols={50}
        />
        <div>
          <button onClick={handleStart} disabled={isRecording}>Start</button>
          <button onClick={handleStop} disabled={!isRecording}>Stop</button>
        </div>
        <p>Transcript: {transcript}</p>
        <video ref={videoRef} autoPlay playsInline muted width="640" height="480" />
      </header>
    </div>
  );
}

export default App;

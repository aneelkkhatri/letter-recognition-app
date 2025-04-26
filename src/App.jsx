import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './LetterRecognition.css';

// Define your custom letters array 
const CUSTOM_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']; 
// You can use any set of characters: Cyrillic, Arabic, Japanese, etc.

// Main component for the letter recognition app
export default function LetterRecognitionApp() {
  const canvasRef = useRef(null);
  const [ctx, setCtx] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [currentLetter, setCurrentLetter] = useState('A');
  const [trainingData, setTrainingData] = useState([]);
  const [message, setMessage] = useState('Draw a letter to recognize or train the model');

  // Initialize canvas and context
  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    context.lineWidth = 20;
    context.lineCap = 'round';
    context.strokeStyle = 'black';
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
    setCtx(context);
    
    // Create and initialize model
    initializeModel();
  }, []);

  // Initialize the TensorFlow.js model
  const initializeModel = async () => {
    try {
      // Create a simple CNN for letter recognition
      const model = tf.sequential();
      
      // Input layer - expects 28x28 grayscale image
      model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
      }));
      model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
      
      model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu'
      }));
      model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
      
      model.add(tf.layers.flatten());
      model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
      model.add(tf.layers.dropout({ rate: 0.2 }));
      
      // Output layer - use the length of your custom letters array
      model.add(tf.layers.dense({ units: CUSTOM_LETTERS.length, activation: 'softmax' }));
      
      model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });
      
      setModel(model);
      setMessage('Model created! Draw a letter or add training data.');
      
    } catch (error) {
      setMessage('Error initializing model: ' + error.message);
    }
  };

  // Handle mouse events for drawing
  const startDrawing = (e) => {
    e.preventDefault(); // Prevent default behavior like scrolling
    const { offsetX, offsetY } = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(offsetX, offsetY);
    setIsDrawing(true);
  };

  const draw = (e) => {
    e.preventDefault(); // Prevent default behavior like scrolling
    if (!isDrawing) return;
    const { offsetX, offsetY } = getCoordinates(e);
    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
  };

  const stopDrawing = (e) => {
    if (e) e.preventDefault(); // Prevent default behavior like scrolling
    if (isDrawing) {
      ctx.closePath();
      setIsDrawing(false);
    }
  };

  // Helper function to get coordinates from both mouse and touch events
  const getCoordinates = (e) => {
    if (e.touches) {
      const rect = canvasRef.current.getBoundingClientRect();
      return {
        offsetX: e.touches[0].clientX - rect.left,
        offsetY: e.touches[0].clientY - rect.top
      };
    }
    return { offsetX: e.nativeEvent.offsetX, offsetY: e.nativeEvent.offsetY };
  };

  // Clear the canvas
  const clearCanvas = () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setPrediction(null);
  };

  // Process the canvas image for the model
  const processCanvasImage = () => {
    return tf.tidy(() => {
      // Get image data from canvas and convert to a tensor
      const imageData = ctx.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height);
      
      // Convert to grayscale tensor
      let tensor = tf.browser.fromPixels(imageData, 1);
      
      // Resize to 28x28 (standard size for many digit/letter recognition models)
      tensor = tf.image.resizeBilinear(tensor, [28, 28]);
      
      // Normalize pixel values
      tensor = tensor.toFloat().div(tf.scalar(255));
      
      // Invert colors (black drawing on white -> white drawing on black)
      tensor = tf.scalar(1).sub(tensor);
      
      // Reshape for the model
      return tensor.expandDims(0);
    });
  };

  // Recognize the drawn letter
  const recognizeLetter = async () => {
    if (!model) {
      setMessage("Model not loaded yet.");
      return;
    }
    
    try {
      const processedImage = processCanvasImage();
      const prediction = await model.predict(processedImage);
      
      // Get the index of the highest probability
      const predictionData = await prediction.data();
      const maxProbability = Math.max(...predictionData);
      const letterIndex = predictionData.indexOf(maxProbability);
      
      // Use the index to get the letter from your custom array
      const predictedLetter = CUSTOM_LETTERS[letterIndex];
      const confidence = (maxProbability * 100).toFixed(2);
      
      setPrediction({ letter: predictedLetter, confidence });
      setMessage(`Prediction: ${predictedLetter} (${confidence}% confidence)`);
      
    } catch (error) {
      setMessage('Error during prediction: ' + error.message);
    }
  };

  // Add current drawing to training data
  const addToTrainingData = () => {
    try {
      const processedImage = processCanvasImage();
      // Get the index of the current letter in your custom array
      const letterIndex = CUSTOM_LETTERS.indexOf(currentLetter);
      
      setTrainingData(prev => [...prev, { 
        image: processedImage, 
        label: letterIndex, 
        letter: currentLetter 
      }]);
      
      setMessage(`Added sample for letter ${currentLetter}. Total samples: ${trainingData.length + 1}`);
      clearCanvas();
    } catch (error) {
      setMessage('Error adding training data: ' + error.message);
    }
  };

  // Train the model with collected data
  const trainModel = async () => {
    if (trainingData.length < 5) {
      setMessage("Add at least 5 samples before training.");
      return;
    }

    setIsTraining(true);
    setMessage("Training model...");

    try {
      // Prepare training data
      const xs = tf.concat(trainingData.map(sample => sample.image));
      
      // Create one-hot encoded labels
      const labels = trainingData.map(sample => sample.label);
      const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), CUSTOM_LETTERS.length);
      
      // Train the model
      await model.fit(xs, ys, {
        epochs: 10,
        batchSize: Math.min(32, trainingData.length),
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            setMessage(`Training... Epoch ${epoch + 1}/10, Loss: ${logs.loss.toFixed(4)}`);
          }
        }
      });
      
      setMessage("Training complete! Try drawing a letter to test.");
    } catch (error) {
      setMessage('Error during training: ' + error.message);
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Letter Recognition App</h1>
      
      <div className="message-box">
        <p className="message-text">{message}</p>
        {prediction && (
          <div className="prediction-box">
            <p className="prediction-letter">Recognized Letter: {prediction.letter}</p>
            <p>Confidence: {prediction.confidence}%</p>
          </div>
        )}
      </div>
      
      <div className="canvas-container">
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className="drawing-canvas"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseLeave={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
          style={{ touchAction: 'none' }}
        />
      </div>
      
      <div className="button-container">
        <button 
          onClick={clearCanvas}
          className="btn btn-red">
          Clear Canvas
        </button>
        <button 
          onClick={recognizeLetter}
          disabled={!model || isTraining}
          className="btn btn-blue">
          Recognize Letter
        </button>
      </div>
      
      <div className="training-panel">
        <h2 className="panel-title">Training Controls</h2>
        <div className="letter-selector">
          <label className="letter-label">Current Letter:</label>
          <select 
            value={currentLetter}
            onChange={(e) => setCurrentLetter(e.target.value)}
            className="select-box">
            {CUSTOM_LETTERS.map(letter => (
              <option key={letter} value={letter}>{letter}</option>
            ))}
          </select>
        </div>
        <div className="training-buttons">
          <button 
            onClick={addToTrainingData}
            disabled={isTraining}
            className="btn btn-green">
            Add Sample
          </button>
          <button 
            onClick={trainModel}
            disabled={isTraining || trainingData.length < 5}
            className="btn btn-purple">
            {isTraining ? "Training..." : "Train Model"}
          </button>
        </div>
        <p className="small-text">
          Training samples: {trainingData.length} 
          {trainingData.length > 0 && (
            <span> ({[...new Set(trainingData.map(d => d.letter))].join(', ')})</span>
          )}
        </p>
      </div>
      
      <div className="footer">
        <p>This app uses a simple neural network to recognize hand-drawn letters.</p>
        <p>For best results, draw centered, clear, uppercase letters and add multiple samples for training.</p>
      </div>
    </div>
  );
}
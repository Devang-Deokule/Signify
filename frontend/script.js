let sentence = "";
let lastPredictionTime = 0;
let delay = 2000; // default delay in milliseconds

// Stability control
let lastDetectedChar = "";
let stableCount = 0;
const requiredStableFrames = 5;

const output = document.getElementById("output");
const sentenceBox = document.getElementById("sentence");
const videoElement = document.getElementById("webcam");
const canvasElement = document.querySelector(".output_canvas");
const canvasCtx = canvasElement.getContext("2d");

// Optional delay slider (if present in HTML)
const delaySlider = document.getElementById("delaySlider");
const delayValue = document.getElementById("delayValue");

if (delaySlider && delayValue) {
  delaySlider.addEventListener("input", () => {
    delay = parseInt(delaySlider.value);
    delayValue.textContent = `${delay} ms`;
  });
}

// MediaPipe Hands setup
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7,
});

hands.onResults(onResults);

// Camera setup
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({ image: videoElement });
  },
  width: 300,
  height: 225,
});

camera.start();

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const now = Date.now();
    if (now - lastPredictionTime > delay) {
      lastPredictionTime = now;

      const landmarks = results.multiHandLandmarks[0]
        .map((landmark) => [landmark.x, landmark.y])
        .flat();

      sendLandmarksToBackend(landmarks);
    }

    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 2 });
      drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 1 });
    }
  }

  canvasCtx.restore();
}

function sendLandmarksToBackend(landmarksRaw) {
  const xs = [], ys = [];

  for (let i = 0; i < landmarksRaw.length; i += 2) {
    xs.push(landmarksRaw[i]);
    ys.push(landmarksRaw[i + 1]);
  }

  const minX = Math.min(...xs);
  const minY = Math.min(...ys);

  const shiftedLandmarks = [];
  for (let i = 0; i < xs.length; i++) {
    shiftedLandmarks.push(xs[i] - minX);
    shiftedLandmarks.push(ys[i] - minY);
  }

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ landmarks: shiftedLandmarks }),
  })
    .then((res) => res.json())
    .then((data) => {
      const prediction = data.prediction;
      output.textContent = prediction;

      if (!["Unknown", "Invalid", "Error"].includes(prediction)) {
        if (prediction === lastDetectedChar) {
          stableCount++;
        } else {
          lastDetectedChar = prediction;
          stableCount = 1;
        }

        if (stableCount >= requiredStableFrames) {
          sentence += prediction;
          sentenceBox.textContent = sentence;

          // Reset stability check to avoid repeats
          stableCount = 0;
          lastDetectedChar = "";
        }
      } else {
        // Reset if hand not stable or bad prediction
        stableCount = 0;
        lastDetectedChar = "";
      }
    })
    .catch((err) => {
      console.error("Prediction error:", err);
    });
}

// Controls
function resetSentence() {
  sentence = "";
  sentenceBox.textContent = "";
}

function backspace() {
  sentence = sentence.slice(0, -1);
  sentenceBox.textContent = sentence;
}

function speakSentence() {
  const synth = window.speechSynthesis;
  const utter = new SpeechSynthesisUtterance(sentence);
  synth.speak(utter);
}

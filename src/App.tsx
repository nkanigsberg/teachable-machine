import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "./App.css";

// const STATUS = document.getElementById("status");
// const VIDEO = document.getElementById("webcam");
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;
const CLASS_NAMES = ["Class 1", "Class 2"];

let mobilenet: tf.GraphModel<string | tf.io.IOHandler> | undefined = undefined;
let gatherDataState = STOP_DATA_GATHER;
// let videoPlaying = false;
let trainingDataInputs: tf.Tensor<tf.Rank>[] = [];
let trainingDataOutputs: number[] = [];
let examplesCount: number[] = [];
let predict = false;

const getModel = () => {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
  );
  model.add(
    tf.layers.dense({ units: CLASS_NAMES.length, activation: "softmax" })
  );

  model.summary();

  // Compile the model with the defined optimizer and specify a loss function to use.
  model.compile({
    // Adam changes the learning rate over time which is useful.
    optimizer: "adam",
    // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
    // Else categoricalCrossentropy is used if more than 2 classes.
    loss:
      CLASS_NAMES.length === 2
        ? "binaryCrossentropy"
        : "categoricalCrossentropy",
    // As this is a classification problem you can record accuracy in the logs too!
    metrics: ["accuracy"],
  });

  return model;
};

function App() {
  const video = useRef<HTMLVideoElement>(null);

  const [status, setStatus] = useState(
    "Loaded TensorFlow.js - version: " + tf.version.tfjs
  );
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [model, setModel] = useState<tf.Sequential>();

  /**
   * Loads the MobileNet model and warms it up so ready for use.
   **/
  async function loadMobileNetFeatureModel() {
    const URL =
      "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";

    mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
    setStatus("MobileNet v3 loaded successfully!");
    // console.log(mobilenet);

    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
      let answer = mobilenet?.predict(
        tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
      );
      console.log((answer as tf.Tensor)?.shape);
      // console.log((answer as tf.Tensor)?.dataSync());
    });
  }

  /** check if the browser supports getUserMedia() by checking for the existence of key browser APIs properties */
  function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  function enableCam() {
    if (hasGetUserMedia()) {
      // getUsermedia parameters.
      const constraints = {
        video: true,
        width: 640,
        height: 480,
      };

      const videoPlayer = video.current;

      if (!videoPlayer) {
        return;
      }
      // Activate the webcam stream.
      navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        videoPlayer.srcObject = stream;
        videoPlayer.addEventListener("loadeddata", function () {
          setVideoPlaying(true);
        });
      });
    } else {
      console.warn("getUserMedia() is not supported by your browser");
    }
  }

  function dataGatherLoop() {
    const videoPlayer = video.current;
    if (videoPlaying && gatherDataState !== STOP_DATA_GATHER && videoPlayer) {
      let imageFeatures = tf.tidy(function () {
        let videoFrameAsTensor = tf.browser.fromPixels(videoPlayer);
        let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor,
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
        );
        let normalizedTensorFrame = resizedTensorFrame.div(255);
        return (
          mobilenet?.predict(normalizedTensorFrame.expandDims()) as tf.Tensor
        ).squeeze();
      });

      trainingDataInputs.push(imageFeatures);
      trainingDataOutputs.push(gatherDataState);

      // Intialize array index element if currently undefined.
      if (examplesCount[gatherDataState] === undefined) {
        examplesCount[gatherDataState] = 0;
      }
      examplesCount[gatherDataState]++;

      let newStatus = "";
      for (let n = 0; n < CLASS_NAMES.length; n++) {
        newStatus += CLASS_NAMES[n] + " data count: " + examplesCount[n] + ". ";
      }
      setStatus(newStatus);
      window.requestAnimationFrame(dataGatherLoop);
    }
  }

  /**
   * Handle Data Gather for button mouseup/mousedown.
   **/
  function gatherDataForClass(e: React.MouseEvent<HTMLButtonElement>) {
    let classNumber = parseInt(
      (e.target as HTMLButtonElement).getAttribute("data-1hot") ?? ""
    );
    gatherDataState =
      gatherDataState === STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;
    dataGatherLoop();
  }

  useEffect(() => {
    loadMobileNetFeatureModel();
    setModel(getModel());
  }, []);

  function predictLoop() {
    const videoPlayer = video.current;
    if (predict && videoPlayer) {
      tf.tidy(function () {
        let videoFrameAsTensor = tf.browser.fromPixels(videoPlayer).div(255);
        let resizedTensorFrame = tf.image.resizeBilinear(
          videoFrameAsTensor as tf.Tensor3D,
          [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
          true
        );

        let imageFeatures = mobilenet?.predict(resizedTensorFrame.expandDims());
        let prediction = (
          model?.predict(imageFeatures as tf.Tensor) as tf.Tensor
        ).squeeze();
        let highestIndex = prediction.argMax().arraySync() as number;
        let predictionArray = prediction.arraySync() as number[];

        setStatus(
          "Prediction: " +
            CLASS_NAMES[highestIndex] +
            " with " +
            Math.floor(predictionArray[highestIndex] * 100) +
            "% confidence"
        );
      });

      window.requestAnimationFrame(predictLoop);
    }
  }

  async function trainAndPredict() {
    predict = false;
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);

    let results = await model?.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true,
      batchSize: 5,
      epochs: 10,
      callbacks: { onEpochEnd: logProgress },
    });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    predict = true;
    predictLoop();
  }

  function logProgress(epoch: number, logs: any) {
    console.log("Data for epoch " + epoch, logs);
  }

  /**
   * Purge data and start over. Note this does not dispose of the loaded
   * MobileNet model and MLP head tensors as you will need to reuse
   * them to train a new model.
   **/
  function reset() {
    predict = false;
    examplesCount.length = 0;
    for (let i = 0; i < trainingDataInputs.length; i++) {
      trainingDataInputs[i].dispose();
    }
    trainingDataInputs.length = 0;
    trainingDataOutputs.length = 0;
    setStatus("No data collected");

    console.log("Tensors in memory: " + tf.memory().numTensors);
  }

  return (
    <div className="App">
      <h1>
        Make your own "Teachable Machine" using Transfer Learning with MobileNet
        v3 in TensorFlow.js using saved graph model from TFHub.
      </h1>
      <p id="status">{status}</p>

      <video ref={video} id="webcam" autoPlay muted></video>

      <button
        id="enableCam"
        onClick={enableCam}
        className={videoPlaying ? "removed" : undefined}
      >
        Enable Webcam
      </button>
      <button
        className="dataCollector"
        data-1hot="0"
        data-name="Class 1"
        onMouseDown={gatherDataForClass}
        onMouseUp={gatherDataForClass}
      >
        Gather Class 1 Data
      </button>
      <button
        className="dataCollector"
        data-1hot="1"
        data-name="Class 2"
        onMouseDown={gatherDataForClass}
        onMouseUp={gatherDataForClass}
      >
        Gather Class 2 Data
      </button>
      <button id="train" onClick={trainAndPredict}>
        Train &amp; Predict!
      </button>
      <button id="reset" onClick={reset}>
        Reset
      </button>
    </div>
  );
}

export default App;

const img = document.getElementById("btn2_img");
const btn2 = document.getElementById("button2");

const btn3 = document.getElementById("button3");

const imgOn  = "assets/icons/REC_STOP_BLEU.png";
const imgOff = "assets/icons/REC_launch_BLEU.png";

let isOn = false;

let isRecording = false;
let mediaRecorder;
let audioChunks = [];

const CHUNK_DURATION = 1 * 10 * 1000;

btn2.addEventListener("click", async (e) => {
  e.preventDefault();

  if (!isRecording) {
    try {
      const baseURL = window.location.origin;
      const startResponse = await fetch(`${baseURL}/start-recording`, {
        method: "POST"
      });
      const startData = await startResponse.json();
      console.log("Start recording response:", startData);

      if (startData.status !== "ok") {
        console.error("Erreur lors du démarrage de l'enregistrement sur le serveur");
        return;
      }
    } catch (err) {
      console.error("Erreur lors de la requête /start-recording:", err);
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = async (event) => {
      if (event.data && event.data.size > 0) {
        const blob = event.data;
        const formData = new FormData();
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        formData.append("file", blob, `chunk_${timestamp}.webm`);

        const isLastChunk = !isRecording;
        formData.append("last_chunk", isLastChunk);

        try {
          const response = await fetch(`${baseURL}/upload-audio`, {
            method: "POST",
            body: formData
          });
          const data = await response.json();
          console.log("Backend response:", data);
        } catch (err) {
          console.error("[btn2.addEventListener] Error while sending file: ", err);
        }
      }
    };

    mediaRecorder.start();

    chunkInterval = setInterval(() => {
      if (mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        mediaRecorder.start();
      }
    }, CHUNK_DURATION);

    isRecording = true;
    setCustomText("J'écoute le cours...");
    img.src = imgOn;

  } else {
    mediaRecorder.stop();
    isRecording = false;
    if (chunkInterval) {
      clearInterval(chunkInterval);
      chunkInterval = null;
    }
    setCustomText("Je réfléchis ...");
    img.src = imgOff;
  }
});

let isRecordingBtn3 = false;
let mediaRecorderBtn3;
let audioChunksBtn3 = [];

btn3.addEventListener("click", async (e) => {
  e.preventDefault();

  if (!isRecordingBtn3) {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderBtn3 = new MediaRecorder(stream);

    audioChunksBtn3 = [];

    mediaRecorderBtn3.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksBtn3.push(event.data);
      }
    };

    mediaRecorderBtn3.onstop = async () => {
      const audioBlob = new Blob(audioChunksBtn3, { type: "audio/webm" });
      const formData = new FormData();
      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      formData.append("file", audioBlob, `full_${timestamp}.webm`);

      try {
        const response = await fetch(`${baseURL}/upload-question`, {
          method: "POST",
          body: formData
        });
        const data = await response.json();
        console.log("Backend response (full record):", data);
      } catch (err) {
        console.error("[btn3] Error while sending file: ", err);
      }
    };

    mediaRecorderBtn3.start();
    isRecordingBtn3 = true;
    setCustomText("J'enregistre...");
    console.log("Recording started (full record mode)");

  } else {
    mediaRecorderBtn3.stop();
    isRecordingBtn3 = false;
    setCustomText("Je réflechis...");
    console.log("Recording stopped (full record mode)");
  }
});
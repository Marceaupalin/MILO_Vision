const baseURL = window.__MILO_BASE_URL__ || (window.location.protocol === 'file:' ? 'http://localhost:5000' : window.location.origin);
const socket = io(baseURL);


let currentAudio = null;

const playBtn = document.getElementById("play-audio-btn");
const button2 = document.getElementById("button2");

document.addEventListener("DOMContentLoaded", () => {
    const savedAudio = localStorage.getItem("lastAudio");
    if (savedAudio) {
        setupAudio(savedAudio);
        playBtn.style.display = "flex";
    }
});


function setupAudio(filename) {
    currentAudio = new Audio(`${baseURL}/get-audio/${filename}`);
    currentAudio.crossOrigin = "anonymous";

    currentAudio.onended = () => {
        setCustomText("Bonjour !");
        playBtn.style.display = "none";
    };
}

socket.on("new_audio", (data) => {
    console.log("Nouvel audio reçu :", data.filename);
    setCustomText("Bonjour !");
    setupAudio(data.filename);

    playBtn.style.display = "flex";
    localStorage.setItem("lastAudio", data.filename);
});

playBtn.addEventListener("click", () => {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        dataArray = new Uint8Array(analyser.frequencyBinCount);
    }

    if (source) source.disconnect();
    source = audioContext.createMediaElementSource(currentAudio);
    source.connect(analyser);
    analyser.connect(audioContext.destination);

    currentAudio.play();
    playBtn.style.display = "none";
});

// Quand on clique sur le bouton 2 pour relancer un enregistrement
button2.addEventListener("click", () => {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        playBtn.style.display = "none";
    }
});

let responseAudio=null

socket.on("new_response_audio", (data) => {
    responseAudio = new Audio(`${baseURL}/get-response-audio/${data.filename}`);
    responseAudio.crossOrigin = "anonymous";
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        dataArray = new Uint8Array(analyser.frequencyBinCount);
    }
    if (source) source.disconnect();
    source = audioContext.createMediaElementSource(responseAudio);
    source.connect(analyser);
    analyser.connect(audioContext.destination);

    console.log("Nouvel audio reçu :", data.filename);
    setCustomText("Je parle...");
    responseAudio.play();

    responseAudio.onended = () => {
        setCustomText("Bonjour !");
    };
});


const button1 = document.getElementById("button1");

button1.addEventListener("click", () => {

    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    if (responseAudio) {
        responseAudio.pause();
        responseAudio.currentTime = 0;
    }
    playBtn.style.display = "none";
    setCustomText("Bonjour !");
});

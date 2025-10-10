const canvas = document.getElementById("particle-sphere");
const ctx = canvas.getContext("2d");
const baseURL = window.location.origin;

canvas.width = 400;
canvas.height = 400;

const numParticles = 150;
const baseRadius = 200;
const tubeRadius = 10;
const circles = [];

const amplitude = 30;
let dynamicAmplitude = amplitude;
const angularFrequency = 3;
const wave_speed = Date.now() * 0.001;

let audioContext;
let analyser;
let dataArray;
let source;

function createCircle( speed, orbitSpeedX, orbitSpeedY) {
    let particles = [];
    for (let j = 0; j < numParticles; j++) {
        const angle = (j / numParticles) * 2 * Math.PI;
        const tubeAngle = Math.random() * 2 * Math.PI;
        particles.push({
            angle,
            tubeAngle,
            angleSpeed: (Math.random() - 0.5) * 0.003
        });
    }
    return {
        particles,
        rotation: 0,
        rotationSpeed: speed,
        orbitRotationX: 0,
        orbitRotationY: 0,
        orbitSpeedX,
        orbitSpeedY
    };
}

circles.push(createCircle(0.0001, -0.002, -0.0015));
circles.push(createCircle(0.0002, 0.0015, 0.001));
circles.push(createCircle(0.0003, 0.001, -0.001));

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

     if (analyser) {
        analyser.getByteTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            let v = (dataArray[i] - 128) / 128;
            sum += v * v;
        }
        let rms = Math.sqrt(sum / dataArray.length);
        dynamicAmplitude = amplitude + rms * 100;
    }

    for (let circle of circles) {
        circle.rotation += circle.rotationSpeed;
        circle.orbitRotationX += circle.orbitSpeedX;
        circle.orbitRotationY += circle.orbitSpeedY;

        for (let p of circle.particles) {
            p.angle += p.angleSpeed;
            const a = p.angle + circle.rotation;

            // Wave deformation
            const deform = Math.sin(a * angularFrequency + wave_speed) * dynamicAmplitude;

            const r = baseRadius + deform;

            // Position initiale
            const tubeX = (r + tubeRadius * Math.cos(p.tubeAngle)) * Math.cos(a);
            const tubeY = (r + tubeRadius * Math.cos(p.tubeAngle)) * Math.sin(a);
            let x = tubeX;
            let y = tubeY;
            let z = tubeRadius * Math.sin(p.tubeAngle);

            // Orbite globale sur x
            let gx = x * Math.cos(circle.orbitRotationY) + z * Math.sin(circle.orbitRotationY);
            let gz = -x * Math.sin(circle.orbitRotationY) + z * Math.cos(circle.orbitRotationY);
            x = gx; z = gz;
            // Orbite globale sur y
            let gy = y * Math.cos(circle.orbitRotationX) - z * Math.sin(circle.orbitRotationX);
            let gz2 = y * Math.sin(circle.orbitRotationX) + z * Math.cos(circle.orbitRotationX);
            y = gy; z = gz2;

            // Projection
            const scale = 200 / (z + baseRadius * 2);
            const x2d = x * scale + centerX;
            const y2d = y * scale + centerY;

            ctx.beginPath();
            ctx.arc(x2d, y2d, 2, 0, 2 * Math.PI);
            ctx.fillStyle = "rgb(0, 0, 128)";
            ctx.fill();
        }
    }

    requestAnimationFrame(draw);
}

draw();
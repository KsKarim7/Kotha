const PLAYBACK_RATE = 16000;
const READY_POLL_MS = 1000;

const consent = document.getElementById("consent");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const status = document.getElementById("status");
const transcript = document.getElementById("transcript");

let call = null;

consent.addEventListener("change", () => {
  startButton.disabled = !consent.checked || call !== null;
});
startButton.addEventListener("click", () => startCall().catch(showFailure));
stopButton.addEventListener("click", () => endCall());

async function waitUntilReady() {
  for (;;) {
    try {
      const response = await fetch("/v1/ready", { cache: "no-store" });
      if (response.ok) {
        return;
      }
    } catch (error) {
      status.textContent = "Waiting for the server…";
    }
    await new Promise((resolve) => setTimeout(resolve, READY_POLL_MS));
  }
}

async function startCall() {
  startButton.disabled = true;
  status.textContent = "Waiting for the server…";
  await waitUntilReady();
  const microphone = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const context = new AudioContext();
  await context.audioWorklet.addModule("/static/pcm-worklet.js");
  const frames = new AudioWorkletNode(context, "pcm-frames");
  context.createMediaStreamSource(microphone).connect(frames);

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/v1/calls/stream`);
  socket.binaryType = "arraybuffer";
  call = { socket, microphone, context, playAt: 0 };

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ type: "start", sample_rate: 16000, consent: true }));
    // frames go out only after start, so the server never refuses early audio
    frames.port.onmessage = (message) => {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(message.data);
      }
    };
    stopButton.disabled = false;
    status.textContent = "Connected. Speak after the greeting.";
  });
  socket.addEventListener("message", (message) => {
    if (message.data instanceof ArrayBuffer) {
      play(message.data);
    } else {
      show(JSON.parse(message.data));
    }
  });
  socket.addEventListener("close", (event) => {
    status.textContent = event.code === 1008 ? "Call refused by the server." : "Call ended.";
    release();
  });
}

function play(buffer) {
  const samples = new Int16Array(buffer);
  const audio = call.context.createBuffer(1, samples.length, PLAYBACK_RATE);
  const channel = audio.getChannelData(0);
  for (let i = 0; i < samples.length; i += 1) {
    channel[i] = samples[i] / 32768;
  }
  const source = call.context.createBufferSource();
  source.buffer = audio;
  source.connect(call.context.destination);
  call.playAt = Math.max(call.playAt, call.context.currentTime);
  source.start(call.playAt);
  call.playAt += audio.duration;
}

function show(event) {
  if (event.type === "vad") {
    status.textContent = event.event === "speech_start" ? "Listening…" : "Thinking…";
  } else if (event.type === "transcript") {
    addLine("caller", `You: ${event.text}`, []);
  } else if (event.type === "bot_text") {
    addLine("bot", `Kotha: ${event.text}`, event.citations.map((citation) => citation.title));
  } else if (event.type === "handoff") {
    status.textContent = "An agent would take over now (demo).";
  } else if (event.type === "reply_audio_end") {
    status.textContent = "Your turn.";
  } else if (event.type === "error") {
    status.textContent = `Server error: ${event.code}`;
  }
}

function addLine(speaker, text, sources) {
  const line = document.createElement("li");
  line.className = speaker;
  line.textContent = text;
  if (sources.length > 0) {
    const note = document.createElement("div");
    note.className = "sources";
    note.textContent = `Sources: ${sources.join(", ")}`;
    line.appendChild(note);
  }
  transcript.appendChild(line);
}

function endCall() {
  if (call && call.socket.readyState === WebSocket.OPEN) {
    call.socket.send(JSON.stringify({ type: "stop" }));
  }
}

function release() {
  if (call === null) {
    return;
  }
  call.microphone.getTracks().forEach((track) => track.stop());
  call.context.close();
  call = null;
  stopButton.disabled = true;
  startButton.disabled = !consent.checked;
}

function showFailure(error) {
  status.textContent = `Could not start the call: ${error.message}`;
  release();
  startButton.disabled = !consent.checked;
}

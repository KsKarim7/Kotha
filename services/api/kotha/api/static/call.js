const PLAYBACK_RATE = 16000;
const READY_POLL_MS = 1000;
const READY_REQUEST_TIMEOUT_MS = 3000;
const READY_MAX_WAIT_MS = 120000;
const STOP_GRACE_MS = 2000;

const consent = document.getElementById("consent");
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const status = document.getElementById("status");
const transcript = document.getElementById("transcript");

let current = null;

class Cancelled extends Error {}

consent.addEventListener("change", () => {
  if (!consent.checked && current !== null) {
    endCall(current);
  }
  updateButtons();
});
startButton.addEventListener("click", () => {
  let context;
  try {
    context = new AudioContext();
  } catch (error) {
    status.textContent = `This browser cannot play or record audio: ${error.message}`;
    return;
  }
  const pending = newCall(context);
  current = pending;
  updateButtons();
  startCall(pending).catch((error) => showFailure(pending, error));
});
stopButton.addEventListener("click", () => {
  if (current !== null) {
    endCall(current);
  }
});

function newCall(context) {
  return {
    context,
    microphone: null,
    socket: null,
    playAt: 0,
    lastError: null,
    sending: false,
    stopTimer: null,
    endedByCaller: false,
    ended: false,
  };
}

function ensureLive(pending) {
  if (pending.ended) {
    release(pending);
    throw new Cancelled();
  }
}

async function waitUntilReady(pending) {
  const giveUpAt = Date.now() + READY_MAX_WAIT_MS;
  while (Date.now() < giveUpAt) {
    ensureLive(pending);
    try {
      const response = await fetch("/v1/ready", {
        cache: "no-store",
        signal: AbortSignal.timeout(READY_REQUEST_TIMEOUT_MS),
      });
      if (response.ok) {
        return;
      }
    } catch (error) {
      if (!pending.ended) {
        status.textContent = "Waiting for the server…";
      }
    }
    await new Promise((resolve) => setTimeout(resolve, READY_POLL_MS));
  }
  throw new Error("the server did not become ready");
}

async function startCall(pending) {
  status.textContent = "Waiting for the server…";
  await waitUntilReady(pending);
  ensureLive(pending);
  pending.microphone = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  ensureLive(pending);
  await pending.context.audioWorklet.addModule("/static/pcm-worklet.js");
  ensureLive(pending);
  const frames = new AudioWorkletNode(pending.context, "pcm-frames", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
  });
  frames.port.onmessage = (message) => {
    if (pending.sending && pending.socket.readyState === WebSocket.OPEN) {
      pending.socket.send(message.data);
    }
  };
  pending.context.createMediaStreamSource(pending.microphone).connect(frames);

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/v1/calls/stream`);
  socket.binaryType = "arraybuffer";
  pending.socket = socket;
  socket.addEventListener("open", () => {
    if (pending.ended || !consent.checked) {
      release(pending);
      return;
    }
    socket.send(JSON.stringify({ type: "start", sample_rate: 16000, consent: true }));
    pending.sending = true;
    updateButtons();
    status.textContent = "Connected. Speak after the greeting.";
  });
  socket.addEventListener("message", (message) => {
    if (pending.ended) {
      return;
    }
    if (message.data instanceof ArrayBuffer) {
      play(pending, message.data);
    } else {
      show(pending, JSON.parse(message.data));
    }
  });
  socket.addEventListener("close", (event) => {
    if (current === pending || current === null) {
      status.textContent = closeMessage(pending, event.code);
    }
    release(pending);
  });
}

function closeMessage(pending, code) {
  if (pending.endedByCaller || code === 1000) {
    return "Call ended.";
  }
  if (code === 1008) {
    return `Call refused by the server (${pending.lastError || "policy"}).`;
  }
  return "Connection lost.";
}

function play(pending, buffer) {
  const samples = new Int16Array(buffer);
  const audio = pending.context.createBuffer(1, samples.length, PLAYBACK_RATE);
  const channel = audio.getChannelData(0);
  for (let i = 0; i < samples.length; i += 1) {
    channel[i] = samples[i] / 32768;
  }
  const source = pending.context.createBufferSource();
  source.buffer = audio;
  source.connect(pending.context.destination);
  pending.playAt = Math.max(pending.playAt, pending.context.currentTime);
  source.start(pending.playAt);
  pending.playAt += audio.duration;
}

function show(pending, event) {
  if (event.type === "vad") {
    status.textContent = event.event === "speech_start" ? "Listening…" : "Thinking…";
  } else if (event.type === "transcript") {
    addLine("caller", `You: ${event.text}`, [], null);
  } else if (event.type === "bot_text") {
    const titles = event.citations.map((citation) => citation.title);
    addLine("bot", `Kotha: ${event.text}`, titles, event.lang);
  } else if (event.type === "handoff") {
    status.textContent = "An agent would take over now (demo).";
  } else if (event.type === "reply_audio_end") {
    status.textContent = "Your turn.";
  } else if (event.type === "error") {
    pending.lastError = event.code;
    status.textContent = `Server error: ${event.code}`;
  }
}

function addLine(speaker, text, sources, lang) {
  const line = document.createElement("li");
  line.className = speaker;
  line.textContent = text;
  if (lang) {
    line.lang = lang;
  }
  if (sources.length > 0) {
    const note = document.createElement("div");
    note.className = "sources";
    note.lang = "en";
    note.textContent = `Sources: ${sources.join(", ")}`;
    line.appendChild(note);
  }
  transcript.appendChild(line);
}

function endCall(pending) {
  pending.sending = false;
  pending.endedByCaller = true;
  if (pending.microphone !== null) {
    pending.microphone.getTracks().forEach((track) => track.stop());
  }
  const socket = pending.socket;
  if (socket === null || socket.readyState !== WebSocket.OPEN || pending.stopTimer !== null) {
    release(pending);
    return;
  }
  socket.send(JSON.stringify({ type: "stop" }));
  pending.stopTimer = setTimeout(() => release(pending), STOP_GRACE_MS);
  updateButtons();
}

function release(pending) {
  pending.ended = true;
  pending.sending = false;
  clearTimeout(pending.stopTimer);
  if (pending.microphone !== null) {
    pending.microphone.getTracks().forEach((track) => track.stop());
  }
  const socket = pending.socket;
  if (socket !== null && socket.readyState <= WebSocket.OPEN) {
    socket.close();
  }
  if (pending.context.state !== "closed") {
    pending.context.close();
  }
  if (current === pending) {
    current = null;
  }
  updateButtons();
}

function updateButtons() {
  startButton.disabled = !consent.checked || current !== null;
  stopButton.disabled = current === null;
}

function showFailure(pending, error) {
  const cancelled = pending.ended || error instanceof Cancelled;
  release(pending);
  if (current === null) {
    status.textContent = cancelled ? "Call cancelled." : `Could not start the call: ${error.message}`;
  }
}

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

let call = null;

consent.addEventListener("change", () => {
  if (!consent.checked && call !== null) {
    endCall();
  }
  startButton.disabled = !consent.checked || call !== null;
});
startButton.addEventListener("click", () => {
  // the call exists from the first click, so a second click or any failure finds it and cleans up
  call = { context: new AudioContext(), microphone: null, socket: null, playAt: 0, lastError: null };
  startButton.disabled = true;
  startCall(call).catch((error) => showFailure(error));
});
stopButton.addEventListener("click", () => endCall());

async function waitUntilReady(pending) {
  const giveUpAt = Date.now() + READY_MAX_WAIT_MS;
  while (Date.now() < giveUpAt) {
    if (call !== pending) {
      throw new Error("the call was cancelled");
    }
    try {
      const response = await fetch("/v1/ready", {
        cache: "no-store",
        signal: AbortSignal.timeout(READY_REQUEST_TIMEOUT_MS),
      });
      if (response.ok) {
        return;
      }
    } catch (error) {
      status.textContent = "Waiting for the server…";
    }
    await new Promise((resolve) => setTimeout(resolve, READY_POLL_MS));
  }
  throw new Error("the server did not become ready");
}

async function startCall(pending) {
  status.textContent = "Waiting for the server…";
  await waitUntilReady(pending);
  pending.microphone = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  await pending.context.audioWorklet.addModule("/static/pcm-worklet.js");
  const frames = new AudioWorkletNode(pending.context, "pcm-frames", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
  });
  let started = false;
  frames.port.onmessage = (message) => {
    if (started && pending.socket.readyState === WebSocket.OPEN) {
      pending.socket.send(message.data);
    }
  };
  pending.context.createMediaStreamSource(pending.microphone).connect(frames);
  if (call !== pending || !consent.checked) {
    throw new Error("the call was cancelled");
  }

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/v1/calls/stream`);
  socket.binaryType = "arraybuffer";
  pending.socket = socket;
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ type: "start", sample_rate: 16000, consent: consent.checked }));
    started = true;
    stopButton.disabled = false;
    status.textContent = "Connected. Speak after the greeting.";
  });
  socket.addEventListener("message", (message) => {
    if (message.data instanceof ArrayBuffer) {
      play(pending, message.data);
    } else {
      show(pending, JSON.parse(message.data));
    }
  });
  socket.addEventListener("close", (event) => {
    status.textContent = closeMessage(event.code, pending.lastError);
    release(pending);
  });
}

function closeMessage(code, lastError) {
  if (code === 1008) {
    return `Call refused by the server (${lastError || "policy"}).`;
  }
  return code === 1000 ? "Call ended." : "Connection lost.";
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

function endCall() {
  const pending = call;
  if (pending === null) {
    return;
  }
  const socket = pending.socket;
  if (socket === null || socket.readyState !== WebSocket.OPEN) {
    release(pending);
    return;
  }
  socket.send(JSON.stringify({ type: "stop" }));
  setTimeout(() => {
    if (socket.readyState !== WebSocket.CLOSED) {
      socket.close();
    }
    release(pending);
  }, STOP_GRACE_MS);
}

function release(pending) {
  if (pending.microphone !== null) {
    pending.microphone.getTracks().forEach((track) => track.stop());
  }
  if (pending.socket !== null && pending.socket.readyState === WebSocket.OPEN) {
    pending.socket.close();
  }
  if (pending.context.state !== "closed") {
    pending.context.close();
  }
  if (call === pending) {
    call = null;
  }
  stopButton.disabled = call === null;
  startButton.disabled = !consent.checked || call !== null;
}

function showFailure(error) {
  status.textContent = `Could not start the call: ${error.message}`;
  if (call !== null) {
    release(call);
  }
}

// Turns microphone audio at the context rate into 20 ms frames of 16 kHz PCM16 (the call protocol).
const TARGET_RATE = 16000;
const FRAME_SAMPLES = 320;

class PcmFrames extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / TARGET_RATE;
    this.position = 0;
    this.previous = 0;
    this.frame = new Int16Array(FRAME_SAMPLES);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0][0];
    if (!channel) {
      return true;
    }
    // linear interpolation; position 0 is the last sample of the previous block
    while (this.position < channel.length) {
      const index = Math.floor(this.position);
      const fraction = this.position - index;
      const before = index === 0 ? this.previous : channel[index - 1];
      const sample = before + (channel[index] - before) * fraction;
      this.push(sample);
      this.position += this.step;
    }
    this.position -= channel.length;
    this.previous = channel[channel.length - 1];
    return true;
  }

  push(sample) {
    const clipped = Math.max(-1, Math.min(1, sample));
    this.frame[this.filled] = Math.round(clipped * 32767);
    this.filled += 1;
    if (this.filled === FRAME_SAMPLES) {
      this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
      this.frame = new Int16Array(FRAME_SAMPLES);
      this.filled = 0;
    }
  }
}

registerProcessor("pcm-frames", PcmFrames);

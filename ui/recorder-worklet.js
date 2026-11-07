// Passes raw microphone samples to app.js, which saves them as an uncompressed WAV.
// Lossy formats from MediaRecorder (Opus, AAC) can alter measures such as jitter and CPP.
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);

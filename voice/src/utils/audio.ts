const PCM_SAMPLE_BYTES = 2;
const STEREO_CHANNELS = 2;

function clamp16(value: number): number {
  if (value > 32767) {
    return 32767;
  }
  if (value < -32768) {
    return -32768;
  }
  return value;
}

export function concatBuffers(buffers: Buffer[]): Buffer {
  if (buffers.length === 0) {
    return Buffer.alloc(0);
  }
  if (buffers.length === 1) {
    return buffers[0];
  }
  return Buffer.concat(buffers);
}

export function downsample48kStereoTo24kMono(pcm48Stereo: Buffer): Buffer {
  if (pcm48Stereo.length === 0) {
    return Buffer.alloc(0);
  }

  const inputSamples = new Int16Array(
    pcm48Stereo.buffer,
    pcm48Stereo.byteOffset,
    Math.floor(pcm48Stereo.byteLength / PCM_SAMPLE_BYTES),
  );

  const frameCount = Math.floor(inputSamples.length / STEREO_CHANNELS);
  const outFrameCount = Math.floor(frameCount / 2);
  const output = Buffer.allocUnsafe(outFrameCount * PCM_SAMPLE_BYTES);

  let outOffset = 0;
  for (let frame = 0; frame < frameCount - 1; frame += 2) {
    const left = inputSamples[frame * 2] ?? 0;
    const right = inputSamples[(frame * 2) + 1] ?? 0;
    const mono = clamp16(Math.round((left + right) / 2));
    output.writeInt16LE(mono, outOffset);
    outOffset += PCM_SAMPLE_BYTES;
  }

  return output.subarray(0, outOffset);
}

export function upsample24kMonoTo48kMono(pcm24Mono: Buffer): Buffer {
  if (pcm24Mono.length === 0) {
    return Buffer.alloc(0);
  }

  const input = new Int16Array(
    pcm24Mono.buffer,
    pcm24Mono.byteOffset,
    Math.floor(pcm24Mono.byteLength / PCM_SAMPLE_BYTES),
  );
  if (input.length === 0) {
    return Buffer.alloc(0);
  }

  const output = Buffer.allocUnsafe(input.length * 2 * PCM_SAMPLE_BYTES);
  let outOffset = 0;

  for (let i = 0; i < input.length; i += 1) {
    const current = input[i] ?? 0;
    const next = input[i + 1] ?? current;
    const midpoint = clamp16(Math.round((current + next) / 2));

    output.writeInt16LE(current, outOffset);
    outOffset += PCM_SAMPLE_BYTES;
    output.writeInt16LE(midpoint, outOffset);
    outOffset += PCM_SAMPLE_BYTES;
  }

  return output.subarray(0, outOffset);
}

export function monoToStereo(pcmMono: Buffer): Buffer {
  if (pcmMono.length === 0) {
    return Buffer.alloc(0);
  }

  const samples = new Int16Array(pcmMono.buffer, pcmMono.byteOffset, Math.floor(pcmMono.byteLength / PCM_SAMPLE_BYTES));
  const output = Buffer.allocUnsafe(samples.length * STEREO_CHANNELS * PCM_SAMPLE_BYTES);

  let outOffset = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const value = samples[i] ?? 0;
    output.writeInt16LE(value, outOffset);
    outOffset += PCM_SAMPLE_BYTES;
    output.writeInt16LE(value, outOffset);
    outOffset += PCM_SAMPLE_BYTES;
  }

  return output.subarray(0, outOffset);
}

export function decodeBase64Pcm(base64: string): Buffer {
  return Buffer.from(base64, "base64");
}

export function encodeBase64Pcm(buffer: Buffer): string {
  return buffer.toString("base64");
}

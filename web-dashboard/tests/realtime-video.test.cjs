const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync, mkdtempSync, writeFileSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const typescript = require('typescript');

const directory = mkdtempSync(join(tmpdir(), 'rsky-video-test-'));
const compiled = join(directory, 'realtime-video.cjs');
writeFileSync(compiled, typescript.transpileModule(readFileSync(join(__dirname, '../lib/realtime-video.ts'), 'utf8'), {
  compilerOptions: { target: typescript.ScriptTarget.ES2022, module: typescript.ModuleKind.CommonJS },
}).outputText);
const { connectRealtimeVideo } = require(compiled);
test.after(() => rmSync(directory, { recursive: true, force: true }));

function fixture(context, options = {}) {
  context.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'], now: 0 });
  context.mock.method(performance, 'now', () => Date.now());
  const peers = [];
  const requests = [];
  const playing = [];
  const resets = [];
  let nextFrame = 0;
  let visible = true;
  const frames = new Map();
  const video = {
    srcObject: null, readyState: 4, paused: false,
    play: context.mock.fn(async () => { video.paused = false; }),
    requestVideoFrameCallback: callback => { frames.set(++nextFrame, callback); return nextFrame; },
    cancelVideoFrameCallback: identifier => frames.delete(identifier),
  };
  if (options.noFrameCallback) delete video.requestVideoFrameCallback;
  class Peer extends EventTarget {
    constructor(configuration) {
      super();
      this.configuration = configuration;
      this.connectionState = 'new';
      this.iceConnectionState = 'new';
      this.iceGatheringState = 'complete';
      this.transceivers = [];
      this.decoded = 0;
      this.receiver = { jitterBufferTarget: null };
      this.track = { kind: 'video', stop: context.mock.fn() };
      this.setRemoteDescription = context.mock.fn(async () => {});
      peers.push(this);
    }
    addTransceiver(kind, configuration) {
      this.transceivers.push({ kind, ...configuration });
      return { receiver: this.receiver };
    }
    async createOffer() { return { type: 'offer', sdp: 'offer-before-gathering' }; }
    async setLocalDescription() { this.localDescription = { sdp: 'offer-with-candidates' }; }
    close() { this.connectionState = 'closed'; this.iceConnectionState = 'closed'; }
    async getStats() {
      return new Map([['video', { type: 'inbound-rtp', kind: 'video', framesDecoded: this.decoded }]]);
    }
    connected() {
      this.connectionState = 'connected';
      this.iceConnectionState = 'connected';
      this.oniceconnectionstatechange?.();
      this.onconnectionstatechange?.();
      this.ontrack?.({ track: this.track, streams: [], receiver: this.receiver });
    }
    state(state) {
      this.connectionState = state;
      this.iceConnectionState = state;
      this.oniceconnectionstatechange?.();
      this.onconnectionstatechange?.();
    }
  }
  class Stream {
    constructor(tracks) { this.tracks = tracks; }
    getTracks() { return this.tracks; }
  }
  const document = new EventTarget();
  document.hidden = false;
  const globals = { RTCPeerConnection: Peer, MediaStream: Stream, document, IntersectionObserver: undefined };
  const restoreGlobals = [];
  for (const [name, value] of Object.entries(globals)) {
    const original = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { value, configurable: true, writable: true });
    restoreGlobals.push(() => {
      if (original) Object.defineProperty(globalThis, name, original);
      else delete globalThis[name];
    });
  }
  const response = () => new Response('answer', { status: 201, headers: { Location: '/camera/whep/session' } });
  context.mock.method(globalThis, 'fetch', async (url, request) => {
    requests.push({ url, ...request });
    if (request.method === 'DELETE') return new Response(null, { status: 200 });
    if (options.post) return options.post(url, request, response);
    return response();
  });
  const stop = connectRealtimeVideo({
    video, url: 'http://localhost:8081/camera/whep', isVisible: () => visible,
    onPlayingChange: value => playing.push(value), onReset: () => resets.push(true),
  });
  context.after(() => { stop(); restoreGlobals.forEach(restore => restore()); });
  const flush = async () => { for (let index = 0; index < 16; index++) await Promise.resolve(); };
  const advance = async milliseconds => {
    for (let elapsed = 0; elapsed < milliseconds; elapsed += 50) {
      context.mock.timers.tick(Math.min(50, milliseconds - elapsed));
      await flush();
    }
  };
  const frame = () => {
    const callbacks = [...frames.values()];
    frames.clear();
    for (const callback of callbacks) callback(performance.now(), {});
  };
  return { peers, requests, playing, resets, video, stop, frame, flush, advance, document, setVisible: value => { visible = value; } };
}

test('only receives video, sends gathered SDP, marks live only after a frame', async context => {
  const check = fixture(context);
  await check.flush();
  const peer = check.peers[0];
  assert.deepEqual(peer.transceivers, [{ kind: 'video', direction: 'recvonly' }]);
  assert.equal(peer.receiver.jitterBufferTarget, 40);
  assert.equal(check.requests[0].body, 'offer-with-candidates');
  peer.connected();
  assert.deepEqual(check.playing, [false]);
  check.frame();
  assert.deepEqual(check.playing, [false, true]);
  assert.equal(check.video.srcObject.getTracks()[0], peer.track);
});

test('one failed POST retries without an iframe or parallel peer', async context => {
  let count = 0;
  const check = fixture(context, { post: async (url, request, response) => ++count === 1 ? new Response('', { status: 503 }) : response() });
  await check.flush();
  assert.equal(check.peers[0].connectionState, 'closed');
  await check.advance(250);
  assert.equal(check.peers.length, 2);
  assert.equal(check.peers.filter(peer => peer.connectionState !== 'closed').length, 1);
  check.peers[1].connected();
  check.frame();
  assert.equal(check.playing.at(-1), true);
});

test('hung negotiation aborts and retries with no accumulation', async context => {
  const check = fixture(context, { post: () => new Promise(() => {}) });
  await check.flush();
  const first = check.requests[0];
  await check.advance(6750);
  assert.equal(first.signal.aborted, true);
  assert.equal(check.peers[0].connectionState, 'closed');
  assert.equal(check.peers.length, 2);
  assert.equal(check.peers.filter(peer => peer.connectionState !== 'closed').length, 1);
});

test('ICE recovery cancels the delayed restart instead of interrupting recovered video', async context => {
  const check = fixture(context);
  await check.flush();
  const peer = check.peers[0];
  peer.connected();
  check.frame();
  peer.state('disconnected');
  await check.advance(500);
  peer.state('connected');
  check.frame();
  await check.advance(750);
  assert.equal(check.peers.length, 1);
  assert.equal(peer.connectionState, 'connected');
});

test('failed peer is closed and stale ontrack cannot replace the new video', async context => {
  const check = fixture(context);
  await check.flush();
  const peer = check.peers[0];
  peer.connected();
  check.frame();
  const oldOnTrack = peer.ontrack;
  peer.state('failed');
  assert.equal(check.video.srcObject, null);
  assert.equal(check.playing.at(-1), false);
  await check.advance(250);
  check.peers[1].connected();
  const stream = check.video.srcObject;
  oldOnTrack({ track: peer.track, receiver: peer.receiver });
  assert.equal(check.video.srcObject, stream);
  assert.equal(peer.track.stop.mock.callCount(), 1);
  assert.equal(check.requests.filter(request => request.method === 'DELETE').length, 1);
});

test('connected but frozen video reconnects without requiring F5', async context => {
  const check = fixture(context);
  await check.flush();
  const peer = check.peers[0];
  peer.connected();
  check.frame();
  await check.advance(2750);
  assert.equal(peer.connectionState, 'closed');
  assert.equal(check.peers.length, 2);
  assert.ok(check.resets.length > 0);
});

test('live decoded frames keep the connection through hidden tabs', async context => {
  const check = fixture(context);
  await check.flush();
  check.peers[0].connected();
  check.frame();
  check.setVisible(false);
  for (let index = 0; index < 12; index++) {
    check.peers[0].decoded += 12;
    await check.advance(500);
  }
  assert.equal(check.peers.length, 1);
  check.setVisible(true);
  await check.advance(500);
  check.frame();
  await check.advance(500);
  assert.equal(check.peers.length, 1);
});

test('visible presentation stall recovers even when RTP continues decoding', async context => {
  const check = fixture(context);
  await check.flush();
  check.peers[0].connected();
  check.frame();
  for (let index = 0; index < 6; index++) {
    check.peers[0].decoded += 12;
    await check.advance(500);
  }
  assert.equal(check.peers[0].connectionState, 'closed');
  assert.equal(check.peers.length, 2);
});

test('background document does not mistake suppressed presentation for a dead stream', async context => {
  const check = fixture(context);
  await check.flush();
  check.peers[0].connected();
  check.frame();
  check.document.hidden = true;
  for (let index = 0; index < 8; index++) {
    check.peers[0].decoded += 12;
    await check.advance(500);
  }
  assert.equal(check.peers.length, 1);
});

test('browser-suspended decoding in a hidden card does not cause reconnect loops', async context => {
  const check = fixture(context);
  await check.flush();
  check.peers[0].connected();
  check.frame();
  check.setVisible(false);
  await check.advance(10000);
  assert.equal(check.peers.length, 1);
  check.setVisible(true);
  await check.advance(500);
  check.frame();
  assert.equal(check.peers.length, 1);
});

test('dispose deletes server session, aborts work, stops tracks and all retries', async context => {
  const check = fixture(context);
  await check.flush();
  check.peers[0].connected();
  check.frame();
  check.stop();
  check.stop();
  await check.advance(15000);
  assert.equal(check.peers.length, 1);
  assert.equal(check.video.srcObject, null);
  assert.equal(check.peers[0].track.stop.mock.callCount(), 1);
  assert.equal(check.requests.filter(request => request.method === 'DELETE').length, 1);
  assert.equal(check.requests[0].signal.aborted, true);
});

test('response after dispose deletes orphan WHEP session without applying SDP', async context => {
  let resolvePost;
  const check = fixture(context, { post: () => new Promise(resolve => { resolvePost = resolve; }) });
  await check.flush();
  check.stop();
  resolvePost(new Response('answer', { status: 201, headers: { Location: '/camera/whep/late-session' } }));
  await check.flush();
  assert.equal(check.peers[0].setRemoteDescription.mock.callCount(), 0);
  assert.equal(check.requests.at(-1).method, 'DELETE');
  assert.equal(check.requests.at(-1).url, 'http://localhost:8081/camera/whep/late-session');
});

test('browsers without frame callback use decoder progress and still detect stalls', async context => {
  const check = fixture(context, { noFrameCallback: true });
  await check.flush();
  check.peers[0].connected();
  check.peers[0].decoded = 12;
  await check.advance(500);
  assert.equal(check.playing.at(-1), true);
  await check.advance(2500);
  assert.equal(check.peers.length, 2);
});

test('repeated HTTP failures back off with only one active attempt', async context => {
  const check = fixture(context, { post: async () => new Response('', { status: 503 }) });
  await check.flush();
  await check.advance(10000);
  assert.ok(check.peers.length <= 7);
  assert.equal(check.peers.filter(peer => peer.connectionState !== 'closed').length, 0);
});

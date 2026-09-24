type RealtimeVideoOptions = {
  video: HTMLVideoElement;
  url: string;
  isVisible: () => boolean;
  onPlayingChange: (playing: boolean) => void;
  onReset: () => void;
};

type VideoAttempt = {
  peer: RTCPeerConnection;
  abort: AbortController;
  stream: MediaStream | null;
  sessionUrl: string | null;
  deadline: ReturnType<typeof setTimeout> | null;
  disconnectTimer: ReturnType<typeof setTimeout> | null;
  frameCallback: number | null;
  startedAt: number;
  decodedAt: number;
  presentedAt: number;
  decodedFrames: number;
  hasFrames: boolean;
  statsPending: boolean;
};

const CONNECT_TIMEOUT_MS = 10000;
const STALL_TIMEOUT_MS = 10000;
const RETRY_MIN_MS = 250;
const RETRY_MAX_MS = 4000;

function preferLowDelay(receiver: RTCRtpReceiver) {
  const tunable = receiver as RTCRtpReceiver & { jitterBufferTarget?: number | null; playoutDelayHint?: number };
  try {
    if (Reflect.has(tunable, 'jitterBufferTarget')) tunable.jitterBufferTarget = 40;
    else if (Reflect.has(tunable, 'playoutDelayHint')) tunable.playoutDelayHint = .04;
  } catch {}
}

function gatherCandidates(peer: RTCPeerConnection, signal: AbortSignal) {
  if (signal.aborted || peer.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise<void>(resolve => {
    const finish = () => {
      clearTimeout(timeout);
      peer.removeEventListener('icegatheringstatechange', changed);
      signal.removeEventListener('abort', finish);
      resolve();
    };
    const changed = () => { if (peer.iceGatheringState === 'complete') finish(); };
    const timeout = setTimeout(finish, 1000);
    peer.addEventListener('icegatheringstatechange', changed);
    signal.addEventListener('abort', finish, { once: true });
  });
}

export function connectRealtimeVideo({ video, url, isVisible, onPlayingChange, onReset }: RealtimeVideoOptions) {
  let stopped = false;
  let current: VideoAttempt | null = null;
  let retryTimer: ReturnType<typeof setTimeout> | null = null;
  let retryDelay = RETRY_MIN_MS;
  let playing = false;
  let inViewport = true;
  let wasVisible = false;

  const setPlaying = (value: boolean) => {
    if (playing === value || stopped) return;
    playing = value;
    onPlayingChange(value);
  };
  const active = (attempt: VideoAttempt) => !stopped && current === attempt;
  const deleteSession = (attempt: VideoAttempt) => {
    const sessionUrl = attempt.sessionUrl;
    attempt.sessionUrl = null;
    if (!sessionUrl) return;
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), 1500);
    void fetch(sessionUrl, { method: 'DELETE', signal: abort.signal, keepalive: true })
      .catch(() => {}).finally(() => clearTimeout(timeout));
  };
  const release = (attempt: VideoAttempt) => {
    if (current === attempt) current = null;
    attempt.abort.abort();
    if (attempt.deadline !== null) clearTimeout(attempt.deadline);
    if (attempt.disconnectTimer !== null) clearTimeout(attempt.disconnectTimer);
    if (attempt.frameCallback !== null) video.cancelVideoFrameCallback(attempt.frameCallback);
    attempt.peer.ontrack = null;
    attempt.peer.oniceconnectionstatechange = null;
    attempt.peer.onconnectionstatechange = null;
    attempt.stream?.getTracks().forEach(track => track.stop());
    attempt.peer.close();
    if (video.srcObject === attempt.stream) {
      video.srcObject = null;
      onReset();
    }
    deleteSession(attempt);
  };
  const retry = (attempt: VideoAttempt) => {
    if (!active(attempt)) return;
    release(attempt);
    setPlaying(false);
    if (retryTimer !== null) return;
    retryTimer = setTimeout(() => {
      retryTimer = null;
      void connect();
    }, retryDelay);
    retryDelay = Math.min(retryDelay * 2, RETRY_MAX_MS);
  };
  const receivedFrames = (attempt: VideoAttempt) => {
    attempt.hasFrames = true;
    if (attempt.deadline !== null) {
      clearTimeout(attempt.deadline);
      attempt.deadline = null;
    }
    if (performance.now() - attempt.startedAt >= 3000) retryDelay = RETRY_MIN_MS;
  };
  const followFrames = (attempt: VideoAttempt) => {
    if (!active(attempt) || typeof video.requestVideoFrameCallback !== 'function') return;
    attempt.frameCallback = video.requestVideoFrameCallback(() => {
      if (!active(attempt)) return;
      attempt.presentedAt = performance.now();
      receivedFrames(attempt);
      setPlaying(true);
      followFrames(attempt);
    });
  };
  const connect = async () => {
    if (stopped || current) return;
    let attempt: VideoAttempt | null = null;
    try {
      const peer = new RTCPeerConnection({ iceServers: [] });
      const startedAt = performance.now();
      attempt = {
        peer, abort: new AbortController(), stream: null, sessionUrl: null,
        deadline: null, disconnectTimer: null, frameCallback: null,
        startedAt, decodedAt: startedAt, presentedAt: startedAt,
        decodedFrames: 0, hasFrames: false, statsPending: false,
      };
      const session = attempt;
      current = session;
      session.deadline = setTimeout(() => retry(session), CONNECT_TIMEOUT_MS);
      const transceiver = peer.addTransceiver('video', { direction: 'recvonly' });
      preferLowDelay(transceiver.receiver);
      peer.ontrack = event => {
        if (!active(session) || event.track.kind !== 'video') return;
        preferLowDelay(event.receiver);
        session.stream = new MediaStream([event.track]);
        video.srcObject = session.stream;
        void video.play().catch(() => {});
      };
      const connectionChanged = () => {
        if (!active(session)) return;
        if (peer.connectionState === 'failed' || peer.connectionState === 'closed' || peer.iceConnectionState === 'failed' || peer.iceConnectionState === 'closed') {
          retry(session);
        } else if (peer.iceConnectionState === 'disconnected' || peer.connectionState === 'disconnected') {
          if (session.disconnectTimer === null) session.disconnectTimer = setTimeout(() => retry(session), 10000);
        } else if (peer.iceConnectionState === 'connected' || peer.iceConnectionState === 'completed') {
          if (session.disconnectTimer !== null) clearTimeout(session.disconnectTimer);
          session.disconnectTimer = null;
        }
      };
      peer.oniceconnectionstatechange = connectionChanged;
      peer.onconnectionstatechange = connectionChanged;
      followFrames(session);
      const offer = await peer.createOffer();
      if (!active(session)) return;
      await peer.setLocalDescription(offer);
      await gatherCandidates(peer, session.abort.signal);
      if (!active(session)) return;
      const response = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/sdp' },
        body: peer.localDescription?.sdp || offer.sdp, signal: session.abort.signal,
      });
      const location = response.headers.get('Location');
      if (location) {
        const resource = new URL(location, url);
        if (resource.origin === new URL(url).origin) session.sessionUrl = resource.href;
      }
      if (!active(session)) {
        deleteSession(session);
        return;
      }
      if (!response.ok) throw new Error(`WHEP HTTP ${response.status}`);
      const sdp = await response.text();
      if (!active(session)) return;
      await peer.setRemoteDescription({ type: 'answer', sdp });
    } catch {
      if (attempt) retry(attempt);
      else if (!stopped && retryTimer === null) {
        retryTimer = setTimeout(() => { retryTimer = null; void connect(); }, retryDelay);
        retryDelay = Math.min(retryDelay * 2, RETRY_MAX_MS);
      }
    }
  };
  const sampleHealth = async (attempt: VideoAttempt) => {
    if (attempt.statsPending) return;
    attempt.statsPending = true;
    try {
      const stats = await attempt.peer.getStats();
      if (!active(attempt)) return;
      let decodedFrames = 0;
      stats.forEach(report => {
        if (report.type === 'inbound-rtp' && (report.kind === 'video' || report.mediaType === 'video')) decodedFrames += report.framesDecoded || 0;
      });
      if (decodedFrames !== attempt.decodedFrames && decodedFrames > 0) {
        attempt.decodedFrames = decodedFrames;
        attempt.decodedAt = performance.now();
        receivedFrames(attempt);
        if (typeof video.requestVideoFrameCallback !== 'function' && video.readyState >= 2) setPlaying(true);
      }
    } catch {} finally { attempt.statsPending = false; }
  };
  const observer = typeof IntersectionObserver === 'function' ? new IntersectionObserver(entries => {
    inViewport = entries.some(entry => entry.isIntersecting);
  }) : null;
  observer?.observe(video);
  const watchdog = setInterval(() => {
    const attempt = current;
    if (!attempt || stopped) return;
    const now = performance.now();
    const visible = !document.hidden && inViewport && isVisible();
    if (visible && !wasVisible) {
      attempt.presentedAt = now;
      void video.play().catch(() => {});
    }
    wasVisible = visible;
    const progressAt = Math.max(attempt.decodedAt, attempt.presentedAt);
    const presentationStalled = visible && typeof video.requestVideoFrameCallback === 'function' && now - attempt.presentedAt > STALL_TIMEOUT_MS;
    if (attempt.hasFrames && visible && (now - progressAt > STALL_TIMEOUT_MS || presentationStalled)) {
      retry(attempt);
      return;
    }
    if (visible && video.paused && video.srcObject) void video.play().catch(() => {});
    void sampleHealth(attempt);
  }, 500);
  const becameVisible = () => {
    wasVisible = false;
    if (!document.hidden && current && video.srcObject) void video.play().catch(() => {});
  };
  document.addEventListener('visibilitychange', becameVisible);
  onPlayingChange(false);
  void connect();
  return () => {
    stopped = true;
    if (retryTimer !== null) clearTimeout(retryTimer);
    clearInterval(watchdog);
    observer?.disconnect();
    document.removeEventListener('visibilitychange', becameVisible);
    if (current) release(current);
  };
}

type RealtimeSocketOptions = {
  url: string;
  onMessage: (event: MessageEvent) => boolean;
  onStatus?: (connected: boolean) => void;
  idleTimeoutMs?: number;
};

export function connectRealtimeSocket({ url, onMessage, onStatus, idleTimeoutMs = 0 }: RealtimeSocketOptions) {
  let disposed = false;
  let socket: WebSocket | null = null;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let retryCount = 0;
  let lastMessageAt = performance.now();
  let connectingAt = lastMessageAt;
  let connected = false;

  const updateStatus = (next: boolean) => {
    if (disposed || connected === next) return;
    connected = next;
    onStatus?.(next);
  };

  const releaseSocket = () => {
    const previous = socket;
    socket = null;
    if (!previous) return;
    previous.onopen = null;
    previous.onmessage = null;
    previous.onerror = null;
    previous.onclose = null;
    try { previous.close(); } catch {}
  };

  const reconnect = () => {
    if (disposed) return;
    releaseSocket();
    updateStatus(false);
    if (reconnectTimer !== null) return;
    const delay = Math.min(250 * 2 ** Math.min(retryCount++, 4), 4000);
    reconnectTimer = setTimeout(connect, delay);
  };

  const connect = () => {
    reconnectTimer = null;
    if (disposed) return;
    releaseSocket();
    connectingAt = performance.now();
    lastMessageAt = connectingAt;
    try {
      const current = new WebSocket(url);
      socket = current;
      current.onopen = () => {
        if (disposed || socket !== current) return;
        lastMessageAt = performance.now();
      };
      current.onmessage = event => {
        if (disposed || socket !== current) return;
        try {
          if (!onMessage(event)) return;
          lastMessageAt = performance.now();
          retryCount = 0;
          updateStatus(true);
        } catch {}
      };
      current.onerror = current.onclose = () => {
        if (!disposed && socket === current) reconnect();
      };
    } catch {
      reconnect();
    }
  };

  const checkHealth = () => {
    if (disposed || !socket) return;
    const now = performance.now();
    if ((socket.readyState === WebSocket.CONNECTING && now - connectingAt > 5000)
      || (socket.readyState === WebSocket.OPEN && idleTimeoutMs > 0 && now - lastMessageAt > idleTimeoutMs)
      || socket.readyState === WebSocket.CLOSED) reconnect();
  };

  const onVisibilityChange = () => {
    if (document.visibilityState === 'visible') checkHealth();
  };

  onStatus?.(false);
  connect();
  const watchdog = setInterval(checkHealth, 500);
  window.addEventListener('online', checkHealth);
  window.addEventListener('focus', checkHealth);
  document.addEventListener('visibilitychange', onVisibilityChange);

  return () => {
    disposed = true;
    if (reconnectTimer !== null) clearTimeout(reconnectTimer);
    clearInterval(watchdog);
    window.removeEventListener('online', checkHealth);
    window.removeEventListener('focus', checkHealth);
    document.removeEventListener('visibilitychange', onVisibilityChange);
    releaseSocket();
  };
}

export function createMetadataClock(maxDelayMs = 500) {
  let minimumOffset: number | null = null;
  return (timestamp: number | undefined, now = Date.now()): number | null => {
    if (!Number.isFinite(timestamp)) return 0;
    const offset = now - timestamp!;
    minimumOffset = minimumOffset === null ? offset : Math.min(minimumOffset, offset);
    const delay = Math.max(0, offset - minimumOffset);
    return delay <= maxDelayMs ? delay : null;
  };
}

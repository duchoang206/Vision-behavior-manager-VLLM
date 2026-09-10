import os
import cv2
import threading
import time
import queue

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class RTSPLatestFrameReader:
    """
    RTSP stream reader using Threading that automatically drops old frames.
    Keeps only the latest frame in memory using a Queue with maxsize=1.
    This resolves the lag when AI processing FPS is lower than Camera FPS.
    """
    def __init__(self, rtsp_url, cam_id="Cam_1", max_reconnect_attempts=-1, decoder_threads=None):
        self.rtsp_url = rtsp_url
        self.cam_id = cam_id
        self.max_reconnect_attempts = max_reconnect_attempts
        self.decoder_threads = decoder_threads
        
        # Initialize video capture
        self.cap = self._open_capture()
        
        # Queue that holds maximum 1 frame
        self.frame_queue = queue.Queue(maxsize=1) 
        self.running = True
        
        # Start the background thread to continuously read frames
        self.thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.thread.start()

    def _open_capture(self):
        if self.decoder_threads is None:
            capture = cv2.VideoCapture(self.rtsp_url)
        else:
            capture = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_N_THREADS, max(1, int(self.decoder_threads)),
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
            ])
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _capture_frames(self):
        attempts = 0
        while self.running:
            if not self.cap.isOpened():
                print(f"[{self.cam_id}] Reconnecting to RTSP stream...")
                self.cap = self._open_capture()
                time.sleep(2)
                
                attempts += 1
                if self.max_reconnect_attempts > 0 and attempts > self.max_reconnect_attempts:
                    print(f"[{self.cam_id}] Max reconnect attempts reached. Stopping thread.")
                    self.running = False
                    break
                continue
            
            # Reset attempts on successful connection check
            attempts = 0

            ret, frame = self.cap.read()
            decoded_at = time.time()
            if not ret:
                print(f"[{self.cam_id}] Failed to read frame or stream ended. Attempting to reconnect...")
                self.cap.release()
                time.sleep(1)
                continue
            
            # If the queue is full, remove the old frame before adding the new one
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Put the latest frame into the queue
            self.frame_queue.put((frame, decoded_at))

    def get_latest_frame(self):
        """
        Returns the most recent frame captured by the background thread.
        Should be called by the AI processing loop.
        
        Returns:
            (bool, np.ndarray): True and the frame if successful, False and None otherwise.
        """
        available, frame, _ = self.get_latest_frame_packet()
        return available, frame

    def get_latest_frame_packet(self):
        try:
            frame, decoded_at = self.frame_queue.get_nowait()
            return True, frame, decoded_at
        except queue.Empty:
            return False, None, None

    def stop(self):
        """
        Gracefully stops the background thread and releases resources.
        """
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print(f"[{self.cam_id}] RTSP Reader stopped successfully.")

if __name__ == "__main__":
    # Example usage / Test block
    print("Testing RTSPLatestFrameReader...")
    # Use 0 for webcam or a valid RTSP URL for testing
    reader = RTSPLatestFrameReader(0, cam_id="Test_Cam")
    
    time.sleep(1) # Allow thread to start and capture first frames
    
    try:
        frames_processed = 0
        start_time = time.time()
        
        while frames_processed < 50: # Test for 50 simulated AI cycles
            ret, frame = reader.get_latest_frame()
            if ret and frame is not None:
                # Simulate AI inference taking 50ms (~20 FPS)
                time.sleep(0.05)
                frames_processed += 1
                if frames_processed % 10 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Processed {frames_processed} frames using latest available.")
            else:
                # If queue is empty, wait a tiny bit
                time.sleep(0.01)
                
        end_time = time.time()
        print(f"Test completed. Simulated AI processed 50 frames in {end_time - start_time:.2f} seconds.")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        reader.stop()

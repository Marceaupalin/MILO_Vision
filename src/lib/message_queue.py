import threading
import time
from collections import defaultdict, deque


class InMemoryEventBus:
    def __init__(self):
        self._queues = defaultdict(deque)
        self._listeners = {}
        self._running = True
        self._lock = threading.Lock()

    def publish(self, stream_name, message_dict):
        print(f"[Producer] {stream_name} -> {message_dict}")
        with self._lock:
            self._queues[stream_name].append(message_dict)

    def subscribe(self, stream_name, listener_name, callback, last_id=None):
        def listener():
            while self._running:
                try:
                    message = None
                    with self._lock:
                        if self._queues[stream_name]:
                            message = self._queues[stream_name].popleft()
                    if message is None:
                        time.sleep(0.1)
                        continue
                    print(f"[{listener_name}] Received on {stream_name}: {message}")
                    callback(message)
                except Exception as e:
                    print(f"[{listener_name}] Error: {e}")
                    time.sleep(0.5)

        t = threading.Thread(target=listener, daemon=True)
        t.start()
        self._listeners[listener_name] = t

    def stop(self):
        self._running = False
        print("Stopping all listeners...")
        for name, t in self._listeners.items():
            t.join(timeout=1)

    def clear_stream(self, stream_name):
        with self._lock:
            self._queues[stream_name].clear()
        print(f"[InMemoryEventBus] Cleared stream: {stream_name}")


message_queue_handler = InMemoryEventBus()

def clearAllStreams():
    message_queue_handler.clear_stream("Audio_topic")
    message_queue_handler.clear_stream("Transcriber_topic")
    message_queue_handler.clear_stream("Question_topic")
    message_queue_handler.clear_stream("Response_topic")
    message_queue_handler.clear_stream("Vision_topic")
import redis
import threading
import time

class RedisEventBus:
    def __init__(self, redis_url="redis://localhost:6379/0"):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.listeners = {}
        self.running = True

    def publish(self, stream_name, message_dict):
        print(f"[Producer] {stream_name} -> {message_dict}")
        self.redis.xadd(stream_name, message_dict)

    def subscribe(self, stream_name, listener_name, callback, last_id="0-0"):
        def listener():
            current_id = last_id
            while self.running:
                try:
                    resp = self.redis.xread({stream_name: current_id}, block=1000, count=10)
                    if not resp:
                        continue
                    for stream, messages in resp:
                        for message_id, message in messages:
                            print(f"[{listener_name}] Received on {stream}: {message}")
                            callback(message)
                            current_id = message_id
                except Exception as e:
                    print(f"[{listener_name}] Error: {e}")
                    time.sleep(1)  # attendre avant retry

        t = threading.Thread(target=listener, daemon=True)
        t.start()
        self.listeners[listener_name] = t

    def stop(self):
        self.running = False
        print("Stopping all Redis listeners...")
        for name, t in self.listeners.items():
            t.join(timeout=1)
    def clear_stream(self, stream_name):
        try:
            self.redis.delete(stream_name)
            print(f"[RedisEventBus] Cleared stream: {stream_name}")
        except Exception as e:
            print(f"[RedisEventBus] Error clearing stream {stream_name}: {e}")


message_queue_handler = RedisEventBus()

def clearAllStreams():
    message_queue_handler.clear_stream("Audio_topic")
    message_queue_handler.clear_stream("Transcriber_topic")
    message_queue_handler.clear_stream("Question_topic")
    message_queue_handler.clear_stream("Response_topic")
    message_queue_handler.clear_stream("Vision_topic")
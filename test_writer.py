import threading
import queue

class MockLogger:
    def log_struct(self, entry, severity, timestamp):
        pass

class _CloudLogger:
    def __init__(self):
        self._background_writes_enabled = True
        self._write_queue = queue.Queue()
        self._writer_stop = threading.Event()
        self._flush_interval_s = 600.0
        self._max_batch_size = 100
        self.logger = MockLogger()
        self.logs_processed = 0

        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_loop(self):
        while not self._writer_stop.is_set():
            self._writer_stop.wait(self._flush_interval_s)
            self._drain_once()
        self._drain_once()

    def _drain_once(self):
        if self._write_queue is None:
            return
        drained = 0
        while drained < self._max_batch_size:
            try:
                entry, ts = self._write_queue.get_nowait()
            except queue.Empty:
                return
            drained += 1
            try:
                self.logger.log_struct(entry, severity='INFO', timestamp=ts)
                self.logs_processed += 1
            except Exception:
                pass

    def flush(self, timeout_s=10.0):
        if self._writer_stop is not None:
            self._writer_stop.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=timeout_s)

l = _CloudLogger()
for i in range(150):
    l._write_queue.put(({}, 0))

l.flush()
print("Logs processed:", l.logs_processed)

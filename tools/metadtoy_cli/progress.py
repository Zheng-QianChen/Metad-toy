"""UI-independent progress events and a throttled terminal renderer.

Callbacks receive dictionaries with phase, completed, total (or None), unit,
elapsed_seconds and eta_seconds. Percentages/ETA refer to the current phase.
Completion is emitted only after output publication succeeds.
"""
import math
import sys
import time


class ProgressReporter:
    def __init__(self, callback=None, interval=.15, clock=time.monotonic):
        self.callback, self.interval, self.clock = callback, interval, clock
        self.started = self.phase_started = clock()
        self.last_emit = -math.inf
        self.phase = None
        self.total = None
        self.unit = ""
        self.completed = 0
        self.timings = {}

    def begin(self, phase, total=None, unit="items", detail=""):
        now = self.clock()
        if self.phase is not None:
            self.timings[self.phase] = self.timings.get(self.phase, 0.) + now - self.phase_started
        self.phase, self.total, self.unit = phase, total, unit
        self.phase_started, self.completed = now, 0
        self.update(0, detail=detail, force=True)

    def update(self, completed, detail="", force=False):
        self.completed = completed
        now = self.clock()
        if self.callback is None or (not force and now - self.last_emit < self.interval):
            return
        elapsed = now - self.phase_started
        eta = None
        if self.total is not None and 0 < completed < self.total and elapsed >= .5:
            eta = elapsed * (self.total - completed) / completed
        event = {"phase": self.phase, "completed": completed, "total": self.total,
                 "unit": self.unit, "elapsed_seconds": elapsed,
                 "total_elapsed_seconds": now - self.started, "eta_seconds": eta,
                 "detail": detail, "status": "running"}
        self.callback(event)
        self.last_emit = now

    def done(self):
        now = self.clock()
        if self.phase is not None:
            self.timings[self.phase] = self.timings.get(self.phase, 0.) + now - self.phase_started
        if self.callback is not None:
            self.callback({"phase": "done", "completed": 1, "total": 1, "unit": "job",
                           "elapsed_seconds": now - self.started, "total_elapsed_seconds": now - self.started,
                           "eta_seconds": 0., "detail": "", "status": "done"})


class TerminalProgress:
    """Render to stderr; stdout remains a machine-readable result stream."""
    labels = {"initialize": "初始化", "hash_inputs": "校验输入", "scan": "扫描轨迹",
              "weights": "重加权", "export": "导出表格", "plot": "生成图片",
              "validate": "复核输入", "publish": "保存结果", "done": "完成"}

    def __init__(self, mode="auto", stream=None):
        if mode not in ("auto", "always", "never"):
            raise ValueError("progress mode must be auto, always or never")
        self.stream = stream if stream is not None else sys.stderr
        self.terminal = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = mode == "always" or (mode == "auto" and self.terminal)
        self.phase = None
        self.width = 0
        self.last_output = -math.inf
        self.closed = False

    def __call__(self, event):
        if not self.enabled or self.closed:
            return
        now = time.monotonic()
        changed = event["phase"] != self.phase
        finished = event["status"] == "done"
        phase_complete = event["total"] is not None and event["completed"] == event["total"]
        if not changed and not finished and not phase_complete and now - self.last_output < (.15 if self.terminal else 1.):
            return
        if changed and self.phase is not None and self.terminal:
            self.stream.write("\n")
            self.width = 0
        self.phase = event["phase"]
        label = self.labels.get(self.phase, self.phase)
        completed, total = event["completed"], event["total"]
        if total is None:
            bar = "[处理中] %s %s" % (completed, event["unit"])
        else:
            fraction = min(1., max(0., completed / total)) if total else 1.
            filled = int(20 * fraction)
            bar = "[" + "=" * filled + " " * (20-filled) + "] %5.1f%%" % (100 * fraction)
            bar += " %s/%s %s" % (completed, total, event["unit"])
        eta = "" if event["eta_seconds"] is None else " 剩余约 %.1fs" % event["eta_seconds"]
        detail = (" " + event["detail"]) if event["detail"] else ""
        text = "%s %s 已用 %.1fs%s%s" % (label, bar, event["elapsed_seconds"], eta, detail)
        if self.terminal:
            self.stream.write("\r" + text + " " * max(0, self.width-len(text)))
            self.width = len(text)
            if finished:
                self.stream.write("\n")
        else:
            self.stream.write(text + "\n")
        self.stream.flush()
        self.last_output = now
        if finished:
            self.closed = True

    def stop(self, message="失败"):
        if self.enabled and not self.closed:
            self.stream.write(("\n" if self.terminal and self.phase else "") + message + "\n")
            self.stream.flush()
            self.closed = True

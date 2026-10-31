"""Fixed-time strategy: ask for each direction in turn, for the same time."""


class FixedTimer:
    def __init__(self, period_s: float = 25):
        self.period_s = period_s

    def decide(self, now_s: float) -> str:
        return "NS" if int(now_s // self.period_s) % 2 == 0 else "EW"

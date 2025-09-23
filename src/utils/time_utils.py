from datetime import time

def parse_hhmm(s: str) -> time:
    h, m = map(int, s.strip().split(":"))
    return time(h, m)

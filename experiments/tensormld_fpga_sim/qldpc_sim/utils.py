from typing import Optional


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("divisor must be > 0")
    return (a + b - 1) // b


def judge_probability(p: float) -> Optional[int]:
    if p > 0.6:
        return 1
    if p < 0.4:
        return 0
    return None

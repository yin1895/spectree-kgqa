"""Measured profile storage and disjoint online timing utilities.

No default costs are fabricated. Real deployment must calibrate the profiled
reader/hardware combination from tuning queries.
"""
from __future__ import annotations
from dataclasses import dataclass
import bisect, json, math
from pathlib import Path
from time import perf_counter


@dataclass
class BinnedFinalCost:
    upper_token_bounds: list[int]
    conservative_ms: list[float]

    @classmethod
    def load(cls, path: str | Path):
        x=json.loads(Path(path).read_text(encoding='utf8'))
        out=cls(x['upper_token_bounds'],x['conservative_ms'])
        if (not out.upper_token_bounds or len(out.upper_token_bounds)!=len(out.conservative_ms)
            or sorted(set(out.upper_token_bounds))!=out.upper_token_bounds
            or any(n<=0 for n in out.upper_token_bounds)
            or any(not math.isfinite(t) or t<0 for t in out.conservative_ms)
            or sorted(out.conservative_ms)!=out.conservative_ms):
            raise ValueError('Bad final-cost lookup')
        return out

    def __call__(self, n_tokens: int) -> float:
        i=bisect.bisect_left(self.upper_token_bounds,n_tokens)
        if i>=len(self.conservative_ms):
            # Profiles outside calibrated range must be calibrated, not silently extrapolated.
            raise ValueError('Token count outside measured cost-profile support')
        return self.conservative_ms[i]


class StageTimer:
    def __init__(self): self.stages={}
    def time(self,stage):
        class Context:
            def __enter__(inner): inner.begin=perf_counter()
            def __exit__(inner,exc_type,exc,tb):
                self.stages[stage]=self.stages.get(stage,0)+(perf_counter()-inner.begin)*1000
        return Context()

    def report(self)->dict:
        return {**self.stages,'mean_stage_total_ms':sum(self.stages.values())}

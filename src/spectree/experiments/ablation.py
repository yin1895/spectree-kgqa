"""Factorial experiment labels and interaction calculation. No outcome fabrication."""
from __future__ import annotations
from itertools import product

ENCODINGS=('flat','dictionary','shared')
SCHEDULES=('fixed','adaptive')


def experiment_grid():
    return [{'encoding':e,'allocation':s, 'name':f'{s.title()}+{e.title()}'}
            for s,e in product(SCHEDULES,ENCODINGS)]


def interaction_f1(aggregates: dict[str,float]) -> float:
    required={'Adaptive+Shared','Fixed+Shared','Adaptive+Dictionary','Fixed+Dictionary'}
    if required-set(aggregates): raise ValueError('Incomplete interaction contrast')
    return ((aggregates['Adaptive+Shared']-aggregates['Fixed+Shared'])
            -(aggregates['Adaptive+Dictionary']-aggregates['Fixed+Dictionary']))

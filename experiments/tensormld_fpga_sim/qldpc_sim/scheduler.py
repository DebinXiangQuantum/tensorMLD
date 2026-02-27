import heapq
from typing import Dict, List, Tuple


def schedule_jobs(job_pairs: List[Tuple[Tuple[int, int], int]], num_engines: int) -> Tuple[int, Dict[Tuple[int, int], int]]:
    """FIFO list scheduling on identical engines.

    job_pairs: [((job_key0, job_key1), duration), ...]
    returns: makespan, completion time per job key
    """
    if not job_pairs:
        return 0, {}
    engines = max(1, num_engines)
    heap = [(0, e) for e in range(engines)]
    heapq.heapify(heap)
    done: Dict[Tuple[int, int], int] = {}
    end_max = 0
    for key, dur in job_pairs:
        start, eid = heapq.heappop(heap)
        finish = start + max(0, dur)
        done[key] = finish
        end_max = max(end_max, finish)
        heapq.heappush(heap, (finish, eid))
    return end_max, done


def schedule_durations(durations: List[int], num_engines: int) -> int:
    if not durations:
        return 0
    jobs = [((i, 0), d) for i, d in enumerate(durations)]
    makespan, _ = schedule_jobs(jobs, num_engines)
    return makespan

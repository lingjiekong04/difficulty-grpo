"""Schedule independent TP1 inference jobs after a serial cold-entry gate."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue


def evaluate_plan(plan, evaluate_item, devices, parallel, public_gate, on_update):
    def key(item):
        return item['variant'], item['seed'], item['step']

    if parallel and (not devices or len(devices) != len(set(devices))):
        raise ValueError('Distinct physical GPU masks required')
    unique = {}
    for item in plan:
        unique.setdefault(key(item), item)
    first = next(iter(unique.values()))
    completed = {key(first): evaluate_item(first, devices[0] if parallel else None,
                                         verify_export=True, public_gate=public_gate)}

    def record():
        rows = [{**item, **completed[key(item)]} for item in plan if key(item) in completed]
        on_update(rows)
        return rows

    results = record()
    remaining = [item for k, item in unique.items() if k not in completed]
    if not parallel:
        for item in remaining:
            completed[key(item)] = evaluate_item(item)
            results = record()
        return results
    pool = Queue()
    for device in devices:
        pool.put(device)

    def worker(item):
        device = pool.get()
        try:
            return evaluate_item(item, device)
        finally:
            pool.put(device)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(worker, item): item for item in remaining}
        try:
            for future in as_completed(futures):
                completed[key(futures[future])] = future.result()
                results = record()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results

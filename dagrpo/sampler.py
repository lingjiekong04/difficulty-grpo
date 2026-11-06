"""Stateful, with-replacement sampling. No cached trajectories."""
import math
import random

class DifficultySampler:
    def __init__(self, ids, variant="A", seed=42, batch_size=32, steps=200, warmup=20):
        if not ids or len(set(ids)) != len(ids) or variant not in ("U", "A", "S", "E"):
            raise ValueError("Unique nonempty IDs and a U/A/S/E variant required")
        self.ids = list(ids)
        self.variant, self.seed = variant, seed
        self.batch_size, self.steps, self.warmup = batch_size, steps, warmup
        self.rng = random.Random(seed)
        self.p = dict.fromkeys(ids, .5)
        self.visits = dict.fromkeys(ids, 0)
        self.completed = self.position = 0

    def probabilities(self):
        n = len(self.ids)
        if self.variant == "U" or self.completed < self.warmup:
            return [1/n] * n
        scores = [self.p[i] * (1-self.p[i]) for i in self.ids]
        total = sum(scores)
        if total == 0:
            return [1/n] * n
        epsilon = 0 if self.variant == "E" else .2
        return [epsilon/n + (1-epsilon)*s/total for s in scores]

    def __len__(self):
        return self.batch_size * self.steps

    def __iter__(self):
        while self.position < len(self):
            index = self.rng.choices(range(len(self.ids)), weights=self.probabilities(), k=1)[0]
            self.position += 1
            yield index

    def feedback(self, groups):
        """groups = [(problem_id, rewards), ...], including repeated occurrences."""
        for pid, rewards in groups:
            if pid not in self.p or not rewards or any(r not in (0, 1) for r in rewards):
                raise ValueError("Unknown problem or nonbinary/empty reward group")
        frozen = self.variant == "S" and self.completed >= self.warmup
        for pid, rewards in groups:
            value = sum(rewards)/len(rewards)
            if not frozen:
                self.p[pid] = value if self.visits[pid] == 0 else .7*self.p[pid] + .3*value
            self.visits[pid] += 1
        self.completed += 1

    def state_dict(self):
        return dict(version=1, ids=self.ids, variant=self.variant, seed=self.seed,
                    batch_size=self.batch_size, steps=self.steps, warmup=self.warmup,
                    p=self.p.copy(), visits=self.visits.copy(), completed=self.completed,
                    position=self.position, rng=self.rng.getstate())

    def load_state_dict(self, state):
        for key in ("ids", "variant", "seed", "batch_size", "steps", "warmup"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Sampler checkpoint mismatch: {key}")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in state["p"].values()):
            raise ValueError("Invalid success estimates")
        self.p, self.visits = state["p"].copy(), state["visits"].copy()
        self.completed, self.position = state["completed"], state["position"]
        def tuples(x):
            return tuple(tuples(y) for y in x) if isinstance(x, (list, tuple)) else x
        self.rng.setstate(tuples(state["rng"]))

def group_feedback(problem_ids, group_ids, rewards, group_size=4):
    groups = {}
    for pid, uid, reward in zip(problem_ids, group_ids, rewards, strict=True):
        if uid not in groups:
            groups[uid] = (pid, [])
        if groups[uid][0] != pid:
            raise ValueError("One rollout group contains different questions")
        groups[uid][1].append(float(reward))
    if any(len(rs) != group_size for _, rs in groups.values()):
        raise ValueError("Incomplete rollout group")
    return list(groups.values())

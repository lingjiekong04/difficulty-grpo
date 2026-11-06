"""Regression for Ray entrypoints with globals detached from the imported module."""
from functools import wraps
import unittest
from dagrpo.train import bind_trainer


class EntrypointTests(unittest.TestCase):
    def test_serialized_traced_entrypoint_uses_research_trainer(self):
        class BaseTrainer:
            def __init__(self, config): self.config = config
            def fit(self): return 'base'

        class ResearchTrainer(BaseTrainer):
            def fit(self): return 'adaptive:' + self.config

        module_globals = {'RayPPOTrainer': BaseTrainer}
        serialized_globals = dict(module_globals)
        exec('def run(self, config):\n return RayPPOTrainer(config).fit()', serialized_globals)
        entrypoint = serialized_globals['run']

        @wraps(entrypoint)
        def traced(*args, **kwargs): return entrypoint(*args, **kwargs)

        module_globals['RayPPOTrainer'] = ResearchTrainer
        self.assertEqual(traced(None, 'A'), 'base')
        self.assertEqual(bind_trainer(traced, ResearchTrainer)(None, 'A'), 'adaptive:A')
        self.assertEqual(traced(None, 'A'), 'base')

    def test_unknown_entrypoint_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'trainer binding missing'):
            bind_trainer(lambda self, config: None, object)


if __name__ == '__main__': unittest.main()

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_backends_complete_finite_training(self):
        from hui.demo import run_demo
        for backend in ('exact', 'checkpoint', 'n2'):
            with self.subTest(backend=backend):
                result = run_demo(backend=backend, steps=3, seed=7)
                self.assertEqual(result['completed_steps'], 3)
                self.assertTrue(torch.isfinite(torch.tensor(result['final_loss'])))
                self.assertEqual(len(result['losses']), 3)

    def test_resume_matches_uninterrupted_model_optimizer_and_data(self):
        from hui.demo import run_demo
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            run_demo('n2', 4, 7, save=p/'full.pt')
            run_demo('n2', 2, 7, save=p/'partial.pt')
            resumed = run_demo('n2', 4, 7, resume=p/'partial.pt', save=p/'resumed.pt')
            full = torch.load(p/'full.pt', weights_only=True)
            actual = torch.load(p/'resumed.pt', weights_only=True)
            self.assertEqual(full['losses'], resumed['losses'])
            for name, tensor in full['model'].items():
                self.assertTrue(torch.equal(tensor, actual['model'][name]), name)
            for key, state in full['optimizer']['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, actual['optimizer']['state'][key][name]), name)
            self.assertTrue(torch.equal(full['data_rng'], actual['data_rng']))

    def test_invalid_steps_and_incompatible_resume_are_rejected(self):
        from hui.demo import run_demo
        with self.assertRaises(ValueError):
            run_demo('exact', 0, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'saved.pt'
            run_demo('n2', 2, 7, save=path)
            for kwargs in ({'backend': 'exact', 'steps': 3, 'seed': 7},
                           {'backend': 'n2', 'steps': 3, 'seed': 8},
                           {'backend': 'n2', 'steps': 1, 'seed': 7}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    run_demo(**kwargs, resume=path)
            with self.assertRaises(FileExistsError):
                run_demo('n2', 2, 7, save=path)

    def test_cli_emits_machine_readable_result(self):
        process = subprocess.run(
            [sys.executable, '-m', 'hui.demo', '--backend', 'checkpoint', '--steps', '2'],
            capture_output=True, text=True, check=True,
        )
        result = json.loads(process.stdout)
        self.assertEqual(result['backend'], 'checkpoint')
        self.assertEqual(result['completed_steps'], 2)

    def test_demo_is_fp32_even_when_callers_default_dtype_is_float64(self):
        from hui.demo import run_demo
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'fp32.pt'
                run_demo('exact', 1, 7, save=path)
                state = torch.load(path, weights_only=True)
                self.assertTrue(all(p.dtype == torch.float32 for p in state['model'].values()))
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(original)


if __name__ == '__main__':
    unittest.main()

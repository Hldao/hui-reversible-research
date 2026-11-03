import copy
import unittest

import torch


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def models(self):
        from hui import build_model, tiny_config
        torch.manual_seed(7)
        exact = build_model(tiny_config(), 'exact')
        models = {'exact': exact}
        for backend in ('checkpoint', 'n2'):
            model = build_model(tiny_config(), backend)
            model.load_state_dict(exact.state_dict())
            models[backend] = model
        return models

    def test_backends_preserve_forward_and_checkpoint_gradients(self):
        models = self.models()
        tokens = torch.arange(32).reshape(2, 16) % 64
        results = {}
        for name, model in models.items():
            logits, loss = model(tokens, labels=tokens)
            loss.backward()
            results[name] = (logits.detach(), torch.cat([p.grad.flatten() for p in model.parameters()]))
        for name in ('checkpoint', 'n2'):
            self.assertTrue(torch.equal(results['exact'][0], results[name][0]), name)
        torch.testing.assert_close(results['checkpoint'][1], results['exact'][1], rtol=1e-5, atol=1e-7)
        error = (results['n2'][1] - results['exact'][1]).norm() / results['exact'][1].norm()
        self.assertLess(float(error), 0.002)

    def test_n2_stores_int8_correction_and_updates_parameters(self):
        from hui import build_model, tiny_config
        model = build_model(tiny_config(), 'n2')
        tokens = torch.arange(16).reshape(1, 16)
        saved_dtypes = []
        def pack(tensor):
            saved_dtypes.append(tensor.dtype)
            return tensor
        before = model.embed_tokens.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            _, loss = model(tokens, labels=tokens)
            loss.backward()
        optimizer.step()
        self.assertIn(torch.int8, saved_dtypes)
        self.assertFalse(torch.equal(before, model.embed_tokens.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_build_does_not_mutate_config_and_clears_legacy_flags(self):
        from hui import build_model, tiny_config
        config = tiny_config()
        config.corrected_reversible_layer_indices = (1,)
        config.checkpoint_reversible_layer_indices = (2,)
        config.reversible_segment_length = 4
        original = copy.deepcopy(config)
        model = build_model(config, 'exact')
        self.assertEqual(config, original)
        self.assertEqual(model.config.corrected_reversible_layer_indices, ())
        self.assertEqual(model.config.checkpoint_reversible_layer_indices, ())
        self.assertEqual(model.config.reversible_segment_length, 0)
        self.assertFalse(model.config.memory_efficient_reversible)

    def test_unknown_backend_and_standard_n2_are_rejected(self):
        from hui import build_model, tiny_config
        with self.assertRaises(ValueError):
            build_model(tiny_config(), 'automatic')
        config = tiny_config()
        config.architecture = 'standard'
        with self.assertRaises(ValueError):
            build_model(config, 'n2')

    def test_bfloat16_n2_gradients_remain_close_to_exact(self):
        models = self.models()
        tokens = torch.arange(32).reshape(2, 16) % 64
        gradients = {}
        for backend in ('exact', 'n2'):
            model = models[backend].to(dtype=torch.bfloat16)
            _, loss = model(tokens, labels=tokens)
            loss.backward()
            gradients[backend] = torch.cat([p.grad.float().flatten() for p in model.parameters()])
            self.assertTrue(torch.isfinite(gradients[backend]).all())
        relative_error = (gradients['n2'] - gradients['exact']).norm() / gradients['exact'].norm()
        cosine = torch.nn.functional.cosine_similarity(gradients['n2'], gradients['exact'], dim=0)
        self.assertLess(float(relative_error), 0.02)
        self.assertGreater(float(cosine), 0.999)

    def test_n2_reports_frozen_reversible_parameters_before_backward(self):
        from hui import build_model, tiny_config
        from hui.torch_module import ReversibleCausalBlock
        model = build_model(tiny_config(), 'n2')
        layer = next(layer for layer in model.layers if isinstance(layer, ReversibleCausalBlock))
        next(layer.parameters()).requires_grad_(False)
        tokens = torch.arange(16).reshape(1, 16)
        with self.assertRaisesRegex(ValueError, 'trainable'):
            model(tokens, labels=tokens)
        # Frozen parameters do not prevent inference, which uses the normal forward.
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(tokens)).all())


if __name__ == '__main__':
    unittest.main()

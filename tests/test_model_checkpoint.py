import unittest

import torch
from torch.utils.checkpoint import checkpoint

from utils.model_checkpoint import checkpoint_model


class ModelCheckpointTests(unittest.TestCase):
    def test_original_nested_output_loses_gradients(self):
        weight = torch.tensor(2.0, requires_grad=True)
        dummy = torch.ones(1, requires_grad=True)

        def forward(unused):
            frames = weight * 3
            return {"positions": frames.square()}, frames

        result = checkpoint(forward, dummy)
        assert isinstance(result, tuple)
        outputs, frames = result
        self.assertFalse(outputs["positions"].requires_grad)
        self.assertTrue(frames.requires_grad)

    def test_nested_only_loss_scales_parameter_gradients(self):
        gradients = []
        for scale in (1.0, 5.0, 20.0):
            weight = torch.tensor(2.0, requires_grad=True)
            dummy = torch.ones(1, requires_grad=True)

            def forward(unused):
                frames = weight * 3
                return {"positions": frames.square()}, frames

            outputs, frames = checkpoint_model(forward, dummy)
            self.assertTrue(outputs["positions"].requires_grad)
            self.assertTrue(frames.requires_grad)
            (scale * outputs["positions"]).backward()
            assert weight.grad is not None
            gradients.append(float(weight.grad))
        self.assertEqual(gradients, [36.0, 180.0, 720.0])

    def test_forward_and_all_gradients_match_direct_with_dropout(self):
        results = []
        for checkpointed in (False, True):
            torch.manual_seed(17)
            model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Dropout(0.2))
            inputs = torch.randn(3, 4, requires_grad=True)

            def forward(value):
                frames = model(value)
                return {"positions": frames.square(), "angles": frames.sin()}, frames

            outputs, frames = (checkpoint_model(forward, inputs) if checkpointed
                               else forward(inputs))
            self.assertEqual(tuple(outputs), ("positions", "angles"))
            loss = outputs["positions"].mean() + 0.4 * outputs["angles"].sum() + frames.square().sum()
            loss.backward()
            gradients = []
            for parameter in model.parameters():
                assert parameter.grad is not None
                gradients.append(parameter.grad.clone())
            assert inputs.grad is not None
            gradients.append(inputs.grad.clone())
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            results.append(([value.detach() for value in (*outputs.values(), frames)], gradients))
        for direct_group, checkpoint_group in zip(results[0], results[1]):
            for direct, checkpointed in zip(direct_group, checkpoint_group):
                self.assertTrue(torch.allclose(direct, checkpointed, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()

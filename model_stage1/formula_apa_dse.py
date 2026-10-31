"""CPU checks for the APA/DSE equations, supervision and learning paths.

Run: python -m unittest discover -s open-r1-multimodal/tests -p test_adsa.py
"""

import math
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "open_r1"))
from modules.adsa import ADSA, APA, DSE, adsa_alignment_loss, apa_alignment_loss


class APAAndDSETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def assert_nonzero_grad(self, tensor):
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all())
        self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_apa_matches_gaussian_pooling_and_cross_attention(self):
        apa = APA(embed_dim=4, text_dim=4)
        tokens = torch.randn(6, 4)
        text = torch.randn(2, 4)
        result = apa(tokens, 2, 3, text)
        expected_centers = torch.tensor([
            [(x + 0.5) / 4, (y + 0.5) / 4] for y in range(4) for x in range(4)
        ])
        torch.testing.assert_close(apa.centers, expected_centers)
        torch.testing.assert_close(apa.scales, torch.full((16, 2), 0.25))
        coords = torch.tensor([[(x + 0.5) / 3, (y + 0.5) / 2] for y in range(2) for x in range(3)])
        g = (-0.5 * ((coords[None] - apa.centers[:, None]) / apa.scales[:, None]).square().sum(-1)).exp()
        a = g / (g.sum(-1, keepdim=True) + apa.eps)
        z = a @ tokens
        alpha = (apa.query_proj(text) @ apa.key_proj(z).t() / math.sqrt(4)).softmax(-1)
        torch.testing.assert_close(result.coordinates, coords)
        torch.testing.assert_close(result.token_association, a)
        torch.testing.assert_close(result.prototype_features, z)
        torch.testing.assert_close(result.prototype_attention, alpha)
        torch.testing.assert_close(result.anatomy_features, apa.output_norm(text + alpha @ apa.value_proj(z)))
        torch.testing.assert_close(result.spatial_maps, alpha @ g)
        # The BCE maps retain Gaussian magnitudes; they are not stretched to [0,1].
        self.assertGreater(result.spatial_maps.min().item(), 0)
        self.assertLess(result.spatial_maps.max().item(), 1)

    def test_bbox_loss_selects_disease_and_averages_instances(self):
        maps = torch.tensor([
            [[0.8, 0.2], [0.3, 0.1]],
            [[0.1, 0.3], [0.2, 0.9]],
        ], requires_grad=True)
        other = torch.full((2, 2), 0.4, requires_grad=True)
        boxes = torch.tensor([
            [0., 0., 0.5, 0.5], [0.5, 0.5, 1., 1.], [0.5, 0., 1., 0.5],
        ])
        ids = torch.tensor([0, 1, 0])
        loss = apa_alignment_loss(
            [maps, other], [(2, 2), (2, 2)], [boxes, torch.tensor([0., 0., 1., 1.])],
            [ids, None],
        )
        targets = torch.tensor([
            [[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]], [[0., 1.], [0., 0.]],
        ])
        expected = torch.stack([
            *[F.binary_cross_entropy(maps[c], target) for c, target in zip(ids, targets)],
            F.binary_cross_entropy(other, torch.ones_like(other)),
        ]).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assert_nonzero_grad(maps)
        self.assert_nonzero_grad(other)
        with self.assertRaisesRegex(ValueError, "category_indices"):
            apa_alignment_loss([maps], [(2, 2)], [boxes])

    def test_tiny_boxes_and_empty_annotations(self):
        response = torch.full((2, 2), 0.3, requires_grad=True)
        tiny_box = torch.tensor([0., 0., 0.01, 0.01])
        loss = adsa_alignment_loss([response], [(2, 2)], [tiny_box])
        torch.testing.assert_close(loss, F.binary_cross_entropy(response, torch.zeros_like(response)))
        loss.backward()
        self.assert_nonzero_grad(response)
        response.grad = None
        zero = apa_alignment_loss([response], [(2, 2)], [torch.empty(0, 4)])
        self.assertEqual(zero.item(), 0)
        zero.backward()
        self.assertIsNotNone(response.grad)
        with self.assertRaisesRegex(ValueError, "normalized"):
            apa_alignment_loss([response], [(2, 2)], [torch.tensor([0., 0., 20., 20.])])

    def test_apa_bbox_gradients_reach_gaussians_attention_and_encoders(self):
        apa = APA(embed_dim=8, text_dim=6, num_prototypes=4)
        tokens = torch.randn(12, 8, requires_grad=True)
        text = torch.randn(3, 6, requires_grad=True)
        result = apa(tokens, 3, 4, text)
        loss = apa_alignment_loss(
            [result.spatial_maps.reshape(3, 3, 4)], [(3, 4)],
            [torch.tensor([[0., 0., 0.5, 0.5], [0.5, 0.5, 1., 1.]])],
            [torch.tensor([0, 2])],
        )
        loss.backward()
        for tensor in [apa.proto_center, apa.proto_raw_sigma, apa.query_proj.weight,
                       apa.key_proj.weight, apa.text_proj.weight, tokens, text]:
            self.assert_nonzero_grad(tensor)
        # L_APA ends at the spatial response; W_val belongs to the grounding path.
        self.assertIsNone(apa.value_proj.weight.grad)

    def test_dse_matches_film_and_feature_gate(self):
        dse = DSE(embed_dim=5, film_hidden=7, identity_init=False)
        tokens = torch.randn(6, 5, requires_grad=True)
        anatomy = torch.randn(3, 5, requires_grad=True)
        actual = dse(tokens, anatomy)
        gamma, beta = dse.film(anatomy).chunk(2, -1)
        modulated = gamma[:, None] * dse.visual_norm(tokens)[None] + beta[:, None]
        concatenated = torch.cat([
            tokens[None].expand(3, -1, -1), modulated,
            anatomy[:, None].expand(-1, 6, -1),
        ], dim=-1)
        expected = tokens[None] + dse.gate(concatenated).sigmoid() * modulated
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        for tensor in [tokens, anatomy, dse.gate.weight, dse.film[0].weight]:
            self.assert_nonzero_grad(tensor)

    def test_identity_initialization_can_learn(self):
        model = ADSA(embed_dim=8, text_dim=6, num_prototypes=4, film_hidden=9)
        tokens = torch.randn(6, 8, requires_grad=True)
        text = torch.randn(6, requires_grad=True)
        grid = torch.tensor([[1, 4, 6]])
        initial = model(tokens, grid, disease_embeds=[text])
        torch.testing.assert_close(initial.image_embeds, tokens, rtol=0, atol=0)
        target = torch.randn_like(tokens)
        F.mse_loss(initial.image_embeds, target).backward()
        self.assert_nonzero_grad(model.dse.film[-1].weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        text.grad = None
        output = model(tokens, grid, disease_embeds=[text])
        self.assertFalse(torch.equal(output.image_embeds, tokens))
        F.mse_loss(output.image_embeds, target).backward()
        for tensor in [text, model.apa.value_proj.weight, model.apa.query_proj.weight,
                       model.apa.proto_center, model.dse.gate.weight]:
            self.assert_nonzero_grad(tensor)

    def test_variable_grids_and_multiple_disease_queries(self):
        model = ADSA(embed_dim=8, text_dim=6, film_hidden=9)
        tokens = torch.randn(8, 8)
        grid = torch.tensor([[1, 4, 6], [1, 2, 4]])
        queries = [torch.randn(3, 6), torch.randn(6)]
        output = model(tokens, grid, disease_embeds=queries)
        self.assertIsNone(output.image_embeds)
        self.assertEqual(output.grids, [(2, 3), (1, 2)])
        self.assertEqual(output.category_image_embeds[0].shape, (3, 6, 8))
        self.assertEqual(output.category_image_embeds[1].shape, (1, 2, 8))
        self.assertEqual(output.saliency[0].shape, (3, 2, 3))
        fallback = model(tokens, grid)
        torch.testing.assert_close(fallback.image_embeds, tokens)
        for result in fallback.apa_outputs:
            torch.testing.assert_close(result.prototype_attention, torch.full((1, 16), 1 / 16))
        for bad_grid in [torch.tensor([[2, 2, 8]]), torch.tensor([[1, 3, 8]]), torch.tensor([[1, 2, 8]])]:
            with self.assertRaises(ValueError):
                model(tokens, bad_grid)
        with self.assertRaisesRegex(ValueError, "per image"):
            model(tokens, grid, disease_embeds=queries[:1])

    def test_bfloat16_geometry_and_backward(self):
        model = ADSA(embed_dim=8, text_dim=6, num_prototypes=4, film_hidden=9).to(torch.bfloat16)
        tokens = torch.randn(6, 8, dtype=torch.bfloat16, requires_grad=True)
        text = torch.randn(2, 6, dtype=torch.bfloat16, requires_grad=True)
        output = model(tokens, torch.tensor([[1, 4, 6]]), disease_embeds=[text])
        self.assertEqual(output.category_image_embeds[0].dtype, tokens.dtype)
        self.assertEqual(output.saliency[0].dtype, torch.float32)
        loss = apa_alignment_loss(
            output.saliency, output.grids, [torch.tensor([[0., 0., 0.5, 1.]])], [torch.tensor([1])],
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assert_nonzero_grad(model.apa.proto_center)
        self.assert_nonzero_grad(text)


if __name__ == "__main__":
    unittest.main()

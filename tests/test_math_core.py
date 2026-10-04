"""Unit tests for LA_ML mathematical core and challenges."""

import unittest
import numpy as np
from app.math_core import (
    compute_transform,
    compute_pca,
    compute_least_squares,
    compute_neural_layer,
    compute_svd,
    compute_low_rank_approx,
    compute_embeddings_similarity,
    compute_vector_analogy,
    compute_svm_classification,
    compute_self_attention,
    compute_hessian_landscape
)
from app.challenges import generate_challenge, verify_challenge


class TestMathCore(unittest.TestCase):
    def test_identity_transform(self):
        res = compute_transform([[1, 0], [0, 1]])
        self.assertEqual(res["det"], 1.0)
        self.assertEqual(res["rank"], 2)
        self.assertFalse(res["is_singular"])
        self.assertEqual(res["transformation_type"], "identity")

    def test_singular_transform(self):
        res = compute_transform([[1, 2], [2, 4]])
        self.assertEqual(res["rank"], 1)
        self.assertTrue(res["is_singular"])
        self.assertEqual(res["condition_number"], "Inf")
        self.assertEqual(res["transformation_type"], "projection_or_singular")

    def test_rotation_transform(self):
        # 90 degree rotation [[0, -1], [1, 0]]
        res = compute_transform([[0, -1], [1, 0]])
        self.assertEqual(res["det"], 1.0)
        self.assertEqual(res["transformation_type"], "rotation")

    def test_pca_basic(self):
        # Points along diagonal y = x
        points = [[float(i), float(i)] for i in range(10)]
        pca = compute_pca(points, n_components=1)
        self.assertAlmostEqual(pca["explained_variance_ratio"][0], 1.0, places=3)
        self.assertAlmostEqual(pca["explained_variance_ratio"][1], 0.0, places=3)
        self.assertAlmostEqual(pca["reconstruction_mse"], 0.0, places=3)

        # 2 components preserves all 2D data
        pca2 = compute_pca([[1.0, 2.0], [3.0, 5.0], [6.0, 2.0]], n_components=2)
        self.assertEqual(pca2["reconstruction_mse"], 0.0)

    def test_least_squares(self):
        # Perfect line y = 2x + 1
        points = [[1.0, 3.0], [2.0, 5.0], [3.0, 7.0], [4.0, 9.0]]
        res = compute_least_squares(points, fit_intercept=True)
        self.assertAlmostEqual(res["slope"], 2.0, places=2)
        self.assertAlmostEqual(res["intercept"], 1.0, places=2)
        self.assertAlmostEqual(res["r2_score"], 1.0, places=2)
        self.assertAlmostEqual(res["orthogonality_norm"], 0.0, places=3)

    def test_neural_layer(self):
        pts = [{"x": 1.0, "y": -1.0, "label": 1}]
        res_relu = compute_neural_layer([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], pts, activation="relu")
        self.assertEqual(res_relu["transformed_points"][0]["activated_a"], [1.0, 0.0])

        res_sig = compute_neural_layer([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [{"x": 0.0, "y": 0.0, "label": 0}], activation="sigmoid")
        self.assertEqual(res_sig["transformed_points"][0]["activated_a"], [0.5, 0.5])

    def test_svd(self):
        A = [[1.5, 1.0], [0.5, 1.5]]
        svd_res = compute_svd(A)
        self.assertEqual(svd_res["rank"], 2)
        self.assertGreater(svd_res["singular_values"][0], svd_res["singular_values"][1])
        # Reconstruct A = U * S * Vt
        U = np.array(svd_res["U"])
        S = np.diag(svd_res["singular_values"])
        Vt = np.array(svd_res["Vt"])
        rec = U @ S @ Vt
        np.testing.assert_allclose(rec, np.array(A), atol=1e-3)

    def test_low_rank_approx(self):
        # Create a 6x6 matrix of rank 1 (outer product)
        u = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = np.array([2.0, 1.0, 0.5, 0.2, 0.1, 0.05])
        matrix = np.outer(u, v).tolist()

        res_k1 = compute_low_rank_approx(matrix, rank_k=1)
        self.assertAlmostEqual(res_k1["energy_retained_pct"], 100.0, places=2)
        self.assertAlmostEqual(res_k1["frobenius_error"], 0.0, places=3)
        self.assertEqual(res_k1["lora_params"], 12)
        self.assertEqual(res_k1["orig_params"], 36)

    def test_embeddings_similarity(self):
        query = [1.0, 0.0]
        docs = [
            {"id": "doc1", "label": "Orthogonal", "vector": [0.0, 1.0]},
            {"id": "doc2", "label": "Same Direction", "vector": [2.0, 0.0]},
            {"id": "doc3", "label": "Opposite", "vector": [-1.0, 0.0]},
        ]
        res = compute_embeddings_similarity(query, docs, top_k=3)
        self.assertEqual(len(res["ranked_documents"]), 3)
        # Top-1 should be 'Same Direction' with cos = 1.0
        self.assertEqual(res["ranked_documents"][0]["id"], "doc2")
        self.assertAlmostEqual(res["ranked_documents"][0]["cosine_similarity"], 1.0, places=3)
        # Second should be doc1 with cos = 0.0
        self.assertEqual(res["ranked_documents"][1]["id"], "doc1")
        self.assertAlmostEqual(res["ranked_documents"][1]["cosine_similarity"], 0.0, places=3)
        # Third should be doc3 with cos = -1.0
        self.assertEqual(res["ranked_documents"][2]["id"], "doc3")
        self.assertAlmostEqual(res["ranked_documents"][2]["cosine_similarity"], -1.0, places=3)

    def test_vector_analogy(self):
        # A - B + C = D
        # King - Man + Woman = Queen
        cand = [
            {"label": "Королева", "vector": [0.5, 3.2]},
            {"label": "Принц", "vector": [1.0, 0.5]},
        ]
        res = compute_vector_analogy(
            vec_a=[2.0, 1.2],
            vec_b=[1.8, -0.6],
            vec_c=[0.3, 1.4],
            vocabulary=cand,
        )
        self.assertEqual(res["top_match"]["label"], "Королева")
        self.assertGreater(res["top_match"]["similarity"], 0.99)

    def test_challenges(self):
        # Vector mapping - ensure non-zero target across 50 generations
        for _ in range(50):
            ch1 = generate_challenge("vector_mapping")
            self.assertIn("u", ch1)
            self.assertIn("target_v", ch1)
            self.assertGreater(abs(ch1["target_v"][0]) + abs(ch1["target_v"][1]), 0.5)

        # Inversion
        ch2 = generate_challenge("matrix_inversion")
        ver = verify_challenge("matrix_inversion", {
            "W": ch2["W"],
            "matrix": ch2["target_inverse"]
        })
        self.assertTrue(ver["success"])

        # SVD Energy Challenge
        ch_svd = generate_challenge("svd_energy")
        self.assertIn("singular_values", ch_svd)
        self.assertIn("correct_rank", ch_svd)
        ver_svd_correct = verify_challenge("svd_energy", {
            "rank": ch_svd["correct_rank"],
            "correct_rank": ch_svd["correct_rank"],
            "cumulative_energies": ch_svd["cumulative_energies"],
        })
        self.assertTrue(ver_svd_correct["success"])

        # RAG Cosine Challenge
        ch_rag = generate_challenge("rag_cosine")
        self.assertIn("target_vector", ch_rag)
        ver_rag_correct = verify_challenge("rag_cosine", {
            "angle_deg": ch_rag["target_angle_deg"],
            "target_angle_deg": ch_rag["target_angle_deg"],
        })
        self.assertTrue(ver_rag_correct["success"])
        self.assertAlmostEqual(ver_rag_correct["cosine_similarity"], 1.0, places=3)

    def test_svm_classification(self):
        # Line x = 0 (w = [1, 0], b = 0)
        # Class +1: (1, 0), (2, 1); Class -1: (-1, 0), (-2, -1)
        pts = [
            {"x": 1.0, "y": 0.0, "label": 1},
            {"x": 2.0, "y": 1.0, "label": 1},
            {"x": -1.0, "y": 0.0, "label": -1},
            {"x": -2.0, "y": -1.0, "label": -1}
        ]
        res = compute_svm_classification(weights=[1.0, 0.0], bias=0.0, points=pts)
        self.assertEqual(res["accuracy_pct"], 100.0)
        self.assertEqual(res["correct_count"], 4)
        self.assertAlmostEqual(res["margin"], 1.0, places=3)
        self.assertAlmostEqual(res["margin_width"], 2.0, places=3)
        self.assertEqual(len(res["support_vector_indices"]), 2)

    def test_self_attention(self):
        tokens = ["A", "B", "C"]
        # Identity query/key embeddings
        Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        V = [[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]
        res = compute_self_attention(tokens, Q, K, V, scale_factor=1.0)
        self.assertEqual(res["n_tokens"], 3)
        self.assertEqual(len(res["attention_map"]), 3)
        # Softmax rows should sum to ~1.0 (allowing float rounding)
        for row in res["attention_map"]:
            self.assertAlmostEqual(sum(row), 1.0, places=3)
        # Output shape N x d_v
        self.assertEqual(len(res["output_embeddings"]), 3)
        self.assertEqual(len(res["output_embeddings"][0]), 2)

    def test_hessian_landscape(self):
        # Condition number 10 (lambda_max = 10.0, lambda_min = 1.0)
        res = compute_hessian_landscape(
            eigenvals=[10.0, 1.0],
            angle_deg=0.0,
            lr=0.05,
            momentum_beta=0.9,
            start_point=[2.0, 2.0],
            steps=20
        )
        self.assertAlmostEqual(res["condition_number"], 10.0, places=2)
        self.assertAlmostEqual(res["lr_max_stable"], 0.2, places=3)
        self.assertFalse(res["is_lr_unstable"])
        self.assertEqual(len(res["trajectory_sgd"]), 21)
        self.assertEqual(len(res["trajectory_momentum"]), 21)
        # Final loss should be lower than initial
        self.assertLess(res["final_loss_momentum"], res["losses_momentum"][0])

    def test_new_challenges(self):
        # 1. Class separation
        ch_sep = generate_challenge("class_separation")
        self.assertEqual(ch_sep["id"], "class_separation")
        self.assertEqual(len(ch_sep["points"]), 16)

        # 2. Invariant eigenvector
        ch_eig = generate_challenge("invariant_eigenvector")
        self.assertEqual(ch_eig["id"], "invariant_eigenvector")
        target_ang = ch_eig["target_angles_deg"][0]
        ver_eig = verify_challenge("invariant_eigenvector", {
            "W": ch_eig["W"],
            "angle_deg": target_ang
        })
        self.assertTrue(ver_eig["success"])

        # 3. Least squares fit
        ch_ls = generate_challenge("least_squares_fit")
        self.assertEqual(ch_ls["id"], "least_squares_fit")
        ver_ls = verify_challenge("least_squares_fit", {
            "points": ch_ls["points"],
            "slope": ch_ls["optimal_slope"],
            "intercept": ch_ls["optimal_intercept"],
            "optimal_residual_norm": ch_ls["optimal_residual_norm"]
        })
        self.assertTrue(ver_ls["success"])

        # 4. RAG Cosine wrap-around angle check (e.g. 10 deg vs 355 deg is 15 deg apart, cos(15 deg) ~ 0.966 >= 0.95)
        ver_rag = verify_challenge("rag_cosine", {
            "angle_deg": 10.0,
            "target_angle_deg": 355.0
        })
        self.assertTrue(ver_rag["success"])

        # 5. Empty points protection
        ver_empty_svm = verify_challenge("class_separation", {"points": []})
        self.assertFalse(ver_empty_svm["success"])
        ver_empty_ls = verify_challenge("least_squares_fit", {"points": []})
        self.assertFalse(ver_empty_ls["success"])


if __name__ == "__main__":
    unittest.main()


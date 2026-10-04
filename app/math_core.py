"""Mathematical core module for Linear Algebra in Machine Learning (LA_ML).

Provides numerical computations for matrix transformations, eigenvalues,
PCA, least squares orthogonal projections, and neural network dense layers.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def compute_transform(matrix: List[List[float]], bias: Optional[List[float]] = None) -> Dict[str, Any]:
    """Compute 2D matrix transformation properties.

    Args:
        matrix: 2x2 matrix [[a, b], [c, d]]
        bias: 2D vector [b1, b2], defaults to [0, 0]

    Returns:
        Dictionary with determinant, rank, eigenvalues, eigenvectors,
        inverse matrix, and transformed basis vectors.
    """
    if bias is None:
        bias = [0.0, 0.0]

    W = np.array(matrix, dtype=float).reshape((2, 2))
    b = np.array(bias, dtype=float).reshape((2,))

    det = float(np.linalg.det(W))
    trace = float(np.trace(W))
    rank = int(np.linalg.matrix_rank(W, tol=1e-5))
    is_singular = abs(det) < 1e-6

    # Inverse
    inverse = None
    if not is_singular:
        try:
            inv_W = np.linalg.inv(W)
            inverse = inv_W.tolist()
        except np.linalg.LinAlgError:
            inverse = None

    # Condition number (for singular matrices, cond is mathematically infinite)
    cond = None
    if not is_singular and rank == 2:
        try:
            val_cond = float(np.linalg.cond(W))
            if not np.isinf(val_cond) and not np.isnan(val_cond) and val_cond < 1e12:
                cond = val_cond
        except Exception:
            cond = None

    # Eigenvalues and Eigenvectors
    eigvals_list: List[Dict[str, Any]] = []
    eigvecs_list: List[List[float]] = []
    try:
        vals, vecs = np.linalg.eig(W)
        for i in range(2):
            val = vals[i]
            vec = vecs[:, i]
            # Check if real
            is_real = bool(abs(val.imag) < 1e-6)
            eigvals_list.append({
                "real": float(val.real),
                "imag": float(val.imag),
                "is_real": is_real,
                "magnitude": float(np.abs(val))
            })
            if is_real:
                # Normalize vector to unit length
                norm = np.linalg.norm(vec.real)
                unit_vec = (vec.real / norm).tolist() if norm > 1e-7 else [0.0, 0.0]
                eigvecs_list.append(unit_vec)
            else:
                eigvecs_list.append([0.0, 0.0])
    except Exception:
        pass

    # Transformed basis vectors: i_hat = [1, 0], j_hat = [0, 1]
    i_hat = np.array([1.0, 0.0])
    j_hat = np.array([0.0, 1.0])
    i_prime = (W @ i_hat + b).tolist()
    j_prime = (W @ j_hat + b).tolist()

    # Unit square deformation: (0,0), (1,0), (1,1), (0,1)
    square_orig = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    square_trans = (square_orig @ W.T + b).tolist()

    # Classification of transformation type
    transformation_type = "general"
    if is_singular:
        transformation_type = "projection_or_singular"
    elif np.allclose(W, np.eye(2), atol=1e-4):
        transformation_type = "identity"
    elif np.allclose(W @ W.T, np.eye(2), atol=1e-4):
        if det > 0:
            transformation_type = "rotation"
        else:
            transformation_type = "reflection"
    elif np.isclose(W[0, 1], 0) and np.isclose(W[1, 0], 0):
        transformation_type = "scaling"
    elif (np.isclose(W[0, 1], 0) and np.isclose(W[0, 0], 1) and np.isclose(W[1, 1], 1)) or \
         (np.isclose(W[1, 0], 0) and np.isclose(W[0, 0], 1) and np.isclose(W[1, 1], 1)):
        transformation_type = "shear"

    return {
        "matrix": W.tolist(),
        "bias": b.tolist(),
        "det": round(det, 4),
        "trace": round(trace, 4),
        "rank": rank,
        "is_singular": is_singular,
        "condition_number": round(cond, 2) if cond is not None else "Inf",
        "inverse": inverse,
        "eigenvalues": eigvals_list,
        "eigenvectors": eigvecs_list,
        "i_prime": i_prime,
        "j_prime": j_prime,
        "square_transformed": square_trans,
        "transformation_type": transformation_type
    }


def compute_pca(points: List[List[float]], n_components: int = 1) -> Dict[str, Any]:
    """Compute Principal Component Analysis (PCA) for 2D points.

    Args:
        points: List of [x, y] coordinates (minimum 2 points).
        n_components: Number of components to keep (1 or 2).

    Returns:
        Dictionary with sample mean, covariance matrix, eigenvalues,
        eigenvectors, explained variance ratios, projections, and MSE.
    """
    pts = np.array(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("Points must be an N x 2 array with at least 2 points.")

    n_samples = pts.shape[0]
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    # Covariance matrix (unbiased: 1 / (N - 1))
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2):
        cov = np.array([[cov, 0.0], [0.0, 0.0]])

    # Eigendecomposition of covariance matrix
    # Since cov is symmetric, np.linalg.eigh is numerically stable
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending by eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx], 0.0)  # Eigenvalues of cov must be >= 0
    eigvecs = eigvecs[:, idx]  # Columns are eigenvectors

    total_var = float(np.sum(eigvals))
    if total_var > 1e-9:
        evr = (eigvals / total_var).tolist()
    else:
        evr = [0.5, 0.5]

    # Principal components
    v1 = eigvecs[:, 0]  # First principal component
    v2 = eigvecs[:, 1]  # Second principal component

    # Projection based on n_components (1 or 2)
    if n_components >= 2:
        t = centered @ eigvecs  # shape (N, 2)
        projected_2d = pts.copy()
        reconstruction_mse = 0.0
    else:
        # 1D projection onto v1
        # Scalar projection: t_i = centered_i . v1
        t = centered @ v1  # shape (N,)
        # Reconstructed 2D points on the PC1 line: mean + t_i * v1
        projected_2d = mean + np.outer(t, v1)
        reconstruction_mse = float(np.mean(np.sum((pts - projected_2d) ** 2, axis=1)))

    # Ellipse parameters for 1-std and 2-std
    # Angle of rotation for the principal component
    angle_rad = float(np.arctan2(v1[1], v1[0]))
    angle_deg = float(np.degrees(angle_rad))

    std_1 = float(np.sqrt(eigvals[0]))
    std_2 = float(np.sqrt(eigvals[1]))

    return {
        "n_samples": n_samples,
        "mean": [round(float(mean[0]), 4), round(float(mean[1]), 4)],
        "cov_matrix": [
            [round(float(cov[0, 0]), 4), round(float(cov[0, 1]), 4)],
            [round(float(cov[1, 0]), 4), round(float(cov[1, 1]), 4)]
        ],
        "eigenvalues": [round(float(eigvals[0]), 4), round(float(eigvals[1]), 4)],
        "eigenvectors": [
            [round(float(v1[0]), 4), round(float(v1[1]), 4)],
            [round(float(v2[0]), 4), round(float(v2[1]), 4)]
        ],
        "explained_variance_ratio": [round(float(evr[0]), 4), round(float(evr[1]), 4)],
        "total_variance": round(total_var, 4),
        "reconstruction_mse": round(reconstruction_mse, 4),
        "ellipse": {
            "angle_deg": round(angle_deg, 2),
            "angle_rad": round(angle_rad, 4),
            "radius_x_1std": round(std_1, 4),
            "radius_y_1std": round(std_2, 4),
            "radius_x_2std": round(std_1 * 2, 4),
            "radius_y_2std": round(std_2 * 2, 4),
        },
        "projected_points": projected_2d.tolist(),
        "scalar_projections": t.tolist()
    }


def compute_least_squares(points: List[List[float]], fit_intercept: bool = True) -> Dict[str, Any]:
    """Compute Linear Regression via Ordinary Least Squares / Subspace Projection.

    Demonstrates y_hat = X(X^T X)^(-1) X^T y as an orthogonal projection of y
    onto the column space Col(X).

    Args:
        points: List of [x, y] coordinates (minimum 2 points).
        fit_intercept: Whether to include bias w0.

    Returns:
        Dictionary with weights, predictions, residuals, projection matrix,
        R^2 score, and geometric projection properties.
    """
    pts = np.array(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
        raise ValueError("At least 2 points (x, y) are required.")

    x_vals = pts[:, 0]
    y_vals = pts[:, 1]
    n = len(x_vals)

    if fit_intercept:
        # Design matrix X: [x, 1]
        X = np.column_stack([x_vals, np.ones(n)])
    else:
        X = x_vals.reshape((-1, 1))

    # Moore-Penrose pseudoinverse: w = (X^T X)^(-1) X^T y
    try:
        X_pinv = np.linalg.pinv(X)
        w = X_pinv @ y_vals
    except np.linalg.LinAlgError:
        raise ValueError("Matrix inversion failed: collinear design matrix.")

    # Projection matrix P = X (X^T X)^(-1) X^T = X @ X_pinv
    P = X @ X_pinv
    y_hat = P @ y_vals
    residuals = y_vals - y_hat  # e = y - y_hat

    # Verify orthogonality: X^T e = 0
    orthogonality_error = float(np.linalg.norm(X.T @ residuals))

    # Metrics
    ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 1.0
    mse = float(np.mean(residuals ** 2))

    slope = float(w[0])
    intercept = float(w[1]) if fit_intercept else 0.0

    # Line endpoints for visualization across x_range
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    span = max(x_max - x_min, 1.0)
    line_x = [x_min - span * 0.2, x_max + span * 0.2]
    line_y = [slope * line_x[0] + intercept, slope * line_x[1] + intercept]

    # Residual segments for visualization
    residual_segments = []
    for i in range(n):
        residual_segments.append({
            "x": float(x_vals[i]),
            "y_actual": float(y_vals[i]),
            "y_pred": float(y_hat[i]),
            "residual": float(residuals[i])
        })

    return {
        "n_samples": n,
        "weights": [round(float(val), 4) for val in w],
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r2_score": round(float(r2), 4),
        "mse": round(mse, 4),
        "orthogonality_norm": round(orthogonality_error, 6),
        "residual_norm": round(float(np.linalg.norm(residuals)), 4),
        "line_coords": {
            "x": line_x,
            "y": line_y
        },
        "residual_segments": residual_segments,
        "projection_matrix_diag": [round(float(P[i, i]), 4) for i in range(min(n, 10))]
    }


def compute_neural_layer(
    matrix: List[List[float]],
    bias: List[float],
    points: List[Dict[str, Any]],
    activation: str = "none"
) -> Dict[str, Any]:
    """Simulate a 2D-to-2D Neural Network Dense Layer: y = sigma(W x + b).

    Demonstrates how matrix weights W rotate/scale the space, bias b shifts it,
    and activation sigma applies non-linear bending to separate classes.
    """
    W = np.array(matrix, dtype=float).reshape((2, 2))
    b = np.array(bias, dtype=float).reshape((2,))

    results = []
    for pt in points:
        x = np.array([pt["x"], pt["y"]], dtype=float)
        # Linear transformation
        z = W @ x + b

        # Non-linear activation
        if activation == "relu":
            a = np.maximum(0.0, z)
        elif activation == "sigmoid":
            a = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        elif activation == "tanh":
            a = np.tanh(z)
        else:  # "none" / linear
            a = z

        results.append({
            "original": [pt["x"], pt["y"]],
            "linear_z": [round(float(z[0]), 4), round(float(z[1]), 4)],
            "activated_a": [round(float(a[0]), 4), round(float(a[1]), 4)],
            "label": pt.get("label", 0)
        })

    return {
        "activation": activation,
        "transformed_points": results
    }


def compute_svd(matrix: List[List[float]]) -> Dict[str, Any]:
    """Compute Singular Value Decomposition A = U Sigma V^T for 2D transformation.

    Deconstructs linear mapping into rotation (V^T), axis scaling (Sigma),
    and final rotation (U), tracking the deformation of the unit circle.
    """
    A = np.array(matrix, dtype=float).reshape((2, 2))
    U, S, Vt = np.linalg.svd(A)
    V = Vt.T

    s1, s2 = float(S[0]), float(S[1])
    rank = int(np.sum(S > 1e-5))
    cond = round(s1 / s2, 2) if s2 > 1e-6 else "Inf"
    frob_norm = float(np.linalg.norm(A, ord='fro'))

    # Unit circle points (48 points)
    thetas = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    circle_pts = np.column_stack([np.cos(thetas), np.sin(thetas)])

    # Stage 0: Unit circle
    pts_stage0 = circle_pts.tolist()
    # Stage 1: Rotated by V^T
    pts_stage1 = (circle_pts @ Vt.T).tolist()
    # Stage 2: Stretched by Sigma
    Sigma = np.diag(S)
    pts_stage2 = (circle_pts @ Vt.T @ Sigma.T).tolist()
    # Stage 3: Final rotation by U -> A x = U Sigma V^T x
    pts_stage3 = (circle_pts @ A.T).tolist()

    return {
        "matrix": A.tolist(),
        "U": [[round(float(x), 4) for x in row] for row in U],
        "Sigma": [round(s1, 4), round(s2, 4)],
        "Vt": [[round(float(x), 4) for x in row] for row in Vt],
        "V": [[round(float(x), 4) for x in row] for row in V],
        "singular_values": [round(s1, 4), round(s2, 4)],
        "u1": [round(float(U[0, 0]), 4), round(float(U[1, 0]), 4)],
        "u2": [round(float(U[0, 1]), 4), round(float(U[1, 1]), 4)],
        "v1": [round(float(V[0, 0]), 4), round(float(V[1, 0]), 4)],
        "v2": [round(float(V[0, 1]), 4), round(float(V[1, 1]), 4)],
        "rank": rank,
        "condition_number": cond,
        "frobenius_norm": round(frob_norm, 4),
        "stages": {
            "circle": pts_stage0,
            "after_vt": pts_stage1,
            "after_sigma": pts_stage2,
            "after_u": pts_stage3
        }
    }


def compute_low_rank_approx(matrix: List[List[float]], rank_k: int = 1) -> Dict[str, Any]:
    """Compute Eckart-Young rank-k SVD approximation and LoRA parameter comparison.

    A_k = sum_{i=1}^k sigma_i u_i v_i^T
    Demonstrates parameter reduction: m*n vs k*(m + n) as used in LoRA fine-tuning.
    """
    A = np.array(matrix, dtype=float)
    if A.ndim != 2:
        raise ValueError("Matrix must be 2-dimensional.")
    m, n = A.shape
    max_rank = min(m, n)
    k = max(1, min(int(rank_k), max_rank))

    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T

    # Rank-k reconstruction: A_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vtk = Vt[:k, :]
    Ak = Uk @ Sk @ Vtk

    total_energy = float(np.sum(S ** 2))
    retained_energy = float(np.sum(S[:k] ** 2))
    energy_ratio = (retained_energy / total_energy) if total_energy > 1e-9 else 1.0

    frob_error = float(np.linalg.norm(A - Ak, ord='fro'))
    orig_params = m * n
    lora_params = k * (m + n)
    compression_percent = round((1.0 - lora_params / orig_params) * 100, 1) if orig_params > 0 else 0.0

    # Components: individual rank-1 matrices sigma_i * u_i * v_i^T
    components = []
    for i in range(min(k, 4)):
        Mi = S[i] * np.outer(U[:, i], Vt[i, :])
        components.append({
            "rank_index": i + 1,
            "singular_value": round(float(S[i]), 4),
            "matrix": [[round(float(val), 3) for val in row] for row in Mi]
        })

    return {
        "original_shape": [m, n],
        "requested_rank": k,
        "max_rank": max_rank,
        "singular_values": [round(float(s), 4) for s in S],
        "approximated_matrix": [[round(float(val), 3) for val in row] for row in Ak],
        "energy_retained_pct": round(energy_ratio * 100, 2),
        "frobenius_error": round(frob_error, 4),
        "orig_params": orig_params,
        "lora_params": lora_params,
        "compression_percent": compression_percent,
        "components": components
    }


def compute_embeddings_similarity(
    query: List[float],
    documents: List[Dict[str, Any]],
    top_k: int = 3
) -> Dict[str, Any]:
    """Compute Cosine Similarity, Euclidean Distance, and Top-K retrieval for RAG.

    cosine_sim(u, v) = (u . v) / (||u|| * ||v||)
    cosine_dist(u, v) = 1 - cosine_sim(u, v)
    """
    q = np.array(query, dtype=float)
    q_norm = float(np.linalg.norm(q))

    results = []
    for doc in documents:
        d = np.array(doc["vector"], dtype=float)
        d_norm = float(np.linalg.norm(d))
        dot_prod = float(np.dot(q, d))

        if q_norm > 1e-7 and d_norm > 1e-7:
            cos_sim = max(-1.0, min(1.0, dot_prod / (q_norm * d_norm)))
            angle_rad = float(np.arccos(cos_sim))
            angle_deg = float(np.degrees(angle_rad))
        else:
            cos_sim = 0.0
            angle_rad = np.pi / 2
            angle_deg = 90.0

        euclidean = float(np.linalg.norm(q - d))
        cos_dist = 1.0 - cos_sim

        results.append({
            "id": doc.get("id", ""),
            "label": doc.get("label", ""),
            "category": doc.get("category", "General"),
            "vector": [round(float(x), 3) for x in d],
            "cosine_similarity": round(cos_sim, 4),
            "cosine_distance": round(cos_dist, 4),
            "angle_deg": round(angle_deg, 1),
            "euclidean_distance": round(euclidean, 4),
            "dot_product": round(dot_prod, 4)
        })

    # Sort descending by cosine similarity
    results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
    for i, res in enumerate(results):
        res["rank"] = i + 1
        res["is_top_k"] = (i < top_k)

    return {
        "query_vector": [round(float(x), 3) for x in q],
        "query_norm": round(q_norm, 4),
        "ranked_documents": results,
        "top_k": results[:top_k]
    }


def compute_vector_analogy(
    vec_a: List[float],
    vec_b: List[float],
    vec_c: List[float],
    vocabulary: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Compute Word2Vec vector analogy: Target = A - B + C (e.g. King - Man + Woman = Queen).

    Finds the closest concept in the vocabulary using Cosine Similarity.
    """
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    c = np.array(vec_c, dtype=float)

    target = a - b + c
    t_norm = float(np.linalg.norm(target))

    matches = []
    for item in (vocabulary or []):
        v = np.array(item["vector"], dtype=float)
        v_norm = float(np.linalg.norm(v))
        if t_norm > 1e-7 and v_norm > 1e-7:
            sim = float(np.dot(target, v) / (t_norm * v_norm))
        else:
            sim = 0.0
        matches.append({
            "label": item.get("label", ""),
            "vector": [round(float(x), 3) for x in v],
            "similarity": round(sim, 4)
        })

    matches.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "result_vector": [round(float(x), 3) for x in target],
        "top_match": matches[0] if matches else None,
        "candidates": matches[:5]
    }


def compute_svm_classification(
    weights: List[float],
    bias: float,
    points: List[Dict[str, Any]],
    canonical_margin: bool = False
) -> Dict[str, Any]:
    """Compute Linear Classification, Hyperplane, and SVM Margin metrics.

    Hyperplane equation: w^T x + b = 0.
    Normal vector w = [w1, w2], distance to plane: d_i = (w^T x_i + b) / ||w||.
    Margin gamma = min_i (y_i (w^T x_i + b)) / ||w||.
    """
    w = np.array(weights, dtype=float)
    b = float(bias)
    norm_w = float(np.linalg.norm(w))

    if norm_w < 1e-6:
        w = np.array([1.0, 0.0])
        norm_w = 1.0

    unit_w = (w / norm_w).tolist()

    evaluated_points = []
    correct_count = 0
    signed_distances = []
    functional_margins = []
    geometric_margins = []

    pos_pts = []
    neg_pts = []

    for idx, pt in enumerate(points):
        x = np.array([pt["x"], pt["y"]], dtype=float)
        label = 1 if pt.get("label", 1) >= 0 else -1
        score = float(w @ x + b)
        dist = score / norm_w
        pred = 1 if score >= 0 else -1
        correct = (pred == label)
        if correct:
            correct_count += 1

        geom_margin = float(label * dist)
        func_margin = float(label * score)

        signed_distances.append(dist)
        functional_margins.append(func_margin)
        geometric_margins.append(geom_margin)

        pt_info = {
            "index": idx,
            "x": float(x[0]),
            "y": float(x[1]),
            "label": label,
            "score": round(score, 4),
            "distance": round(dist, 4),
            "prediction": pred,
            "correct": correct,
            "geometric_margin": round(geom_margin, 4)
        }
        evaluated_points.append(pt_info)

        if label == 1:
            pos_pts.append(pt_info)
        else:
            neg_pts.append(pt_info)

    total = len(points)
    accuracy_pct = round((correct_count / total * 100.0) if total > 0 else 0.0, 2)

    # Calculate SVM Margin (distance to closest points from decision boundary)
    correct_geom_margins = [pt["geometric_margin"] for pt in evaluated_points if pt["correct"]]
    margin_val = min(correct_geom_margins) if correct_geom_margins else 0.0
    margin_val = max(0.0, margin_val)

    # Support vectors: points with minimal geometric margin in each class
    pos_sorted = sorted([p for p in pos_pts if p["correct"]], key=lambda p: abs(p["distance"])) if pos_pts else []
    neg_sorted = sorted([p for p in neg_pts if p["correct"]], key=lambda p: abs(p["distance"])) if neg_pts else []

    support_vector_indices = []
    if pos_sorted:
        support_vector_indices.append(pos_sorted[0]["index"])
    if neg_sorted:
        support_vector_indices.append(neg_sorted[0]["index"])

    for p in evaluated_points:
        p["is_support_vector"] = p["index"] in support_vector_indices

    # Hyperplane line coordinates across range [-10, 10]
    # w1 * x + w2 * y + b = 0  =>  if w2 != 0: y = (-w1 * x - b) / w2
    #                           if w2 == 0: x = -b / w1
    x_span = [-8.0, 8.0]
    y_span = [-8.0, 8.0]

    def get_line_pts(offset: float):
        # w1 * x + w2 * y + b + offset = 0
        w1, w2 = w[0], w[1]
        c = b + offset
        if abs(w2) > abs(w1):
            x1, x2 = x_span[0], x_span[1]
            y1 = (-w1 * x1 - c) / w2
            y2 = (-w1 * x2 - c) / w2
            return [[round(x1, 3), round(y1, 3)], [round(x2, 3), round(y2, 3)]]
        else:
            y1, y2 = y_span[0], y_span[1]
            x1 = (-w2 * y1 - c) / w1
            x2 = (-w2 * y2 - c) / w1
            return [[round(x1, 3), round(y1, 3)], [round(x2, 3), round(y2, 3)]]

    # Separating line: w^T x + b = 0
    main_line = get_line_pts(0.0)

    # Margin lines (at distance gamma: w^T x + b = +- gamma * ||w|| or +- 1)
    offset_dist = margin_val * norm_w if margin_val > 0.01 else 1.0
    pos_margin_line = get_line_pts(-offset_dist)
    neg_margin_line = get_line_pts(offset_dist)

    # Point on hyperplane closest to origin: x0 = -b * w / ||w||^2
    origin_proj = (-b * w / (norm_w ** 2)).tolist()
    normal_end = (np.array(origin_proj) + (w / norm_w) * 1.5).tolist()

    # Hinge loss: sum(max(0, 1 - y_i (w^T x_i + b)))
    hinge_losses = [max(0.0, 1.0 - pt["label"] * (w @ [pt["x"], pt["y"]] + b)) for pt in points]
    mean_hinge = float(np.mean(hinge_losses)) if hinge_losses else 0.0

    return {
        "weights": [round(float(w[0]), 4), round(float(w[1]), 4)],
        "bias": round(b, 4),
        "norm_w": round(norm_w, 4),
        "unit_normal": [round(unit_w[0], 4), round(unit_w[1], 4)],
        "accuracy_pct": accuracy_pct,
        "correct_count": correct_count,
        "total_count": total,
        "margin": round(margin_val, 4),
        "margin_width": round(margin_val * 2.0, 4),
        "mean_hinge_loss": round(mean_hinge, 4),
        "hyperplane_line": main_line,
        "positive_margin_line": pos_margin_line,
        "negative_margin_line": neg_margin_line,
        "normal_origin": [round(origin_proj[0], 4), round(origin_proj[1], 4)],
        "normal_vector_end": [round(normal_end[0], 4), round(normal_end[1], 4)],
        "points": evaluated_points,
        "support_vector_indices": support_vector_indices
    }


def compute_self_attention(
    tokens: List[str],
    Q: List[List[float]],
    K: List[List[float]],
    V: List[List[float]],
    scale_factor: Optional[float] = None
) -> Dict[str, Any]:
    """Compute Scaled Dot-Product Self-Attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Demonstrates matrix-based information routing across Transformer tokens.
    """
    Q_arr = np.array(Q, dtype=float)
    K_arr = np.array(K, dtype=float)
    V_arr = np.array(V, dtype=float)

    n_tokens = len(tokens)
    if Q_arr.shape[0] != n_tokens or K_arr.shape[0] != n_tokens or V_arr.shape[0] != n_tokens:
        raise ValueError("Number of rows in Q, K, V must match length of tokens list.")

    d_k = Q_arr.shape[1]
    d_v = V_arr.shape[1]

    if scale_factor is None or scale_factor <= 0:
        scale_factor = 1.0 / np.sqrt(max(1, d_k))

    # 1. Raw dot products: S = Q @ K^T (N x N)
    Kt_arr = K_arr.T
    raw_dot = Q_arr @ Kt_arr

    # 2. Scaled scores: S_scaled = S / sqrt(d_k)
    scaled_scores = raw_dot * scale_factor

    # 3. Softmax per row: A_ij = exp(S_ij - max_i) / sum_m exp(S_im - max_i)
    attention_weights = np.zeros_like(scaled_scores)
    for i in range(n_tokens):
        row = scaled_scores[i]
        shift = row - np.max(row)
        exps = np.exp(shift)
        attention_weights[i] = exps / np.sum(exps)

    # 4. Output: O = Attention @ V (N x d_v)
    output = attention_weights @ V_arr

    # Step-by-step arithmetic trace for each token query
    token_traces = []
    for i in range(n_tokens):
        dot_row = [round(float(x), 4) for x in raw_dot[i]]
        scaled_row = [round(float(x), 4) for x in scaled_scores[i]]
        attn_row = [round(float(x), 4) for x in attention_weights[i]]
        out_row = [round(float(x), 4) for x in output[i]]

        token_traces.append({
            "token": tokens[i],
            "query_index": i,
            "query_vector": [round(float(x), 3) for x in Q_arr[i]],
            "dot_products": dot_row,
            "scaled_scores": scaled_row,
            "attention_distribution": attn_row,
            "output_vector": out_row,
            "top_attended_token": tokens[int(np.argmax(attention_weights[i]))],
            "top_attention_weight": round(float(np.max(attention_weights[i])), 4)
        })

    return {
        "tokens": tokens,
        "n_tokens": n_tokens,
        "d_k": d_k,
        "d_v": d_v,
        "scale_factor": round(float(scale_factor), 4),
        "Q": [[round(float(x), 3) for x in row] for row in Q_arr],
        "K": [[round(float(x), 3) for x in row] for row in K_arr],
        "Kt": [[round(float(x), 3) for x in row] for row in Kt_arr],
        "V": [[round(float(x), 3) for x in row] for row in V_arr],
        "raw_dot_products": [[round(float(x), 3) for x in row] for row in raw_dot],
        "scaled_scores": [[round(float(x), 3) for x in row] for row in scaled_scores],
        "attention_map": [[round(float(x), 4) for x in row] for row in attention_weights],
        "output_embeddings": [[round(float(x), 4) for x in row] for row in output],
        "token_traces": token_traces
    }


def compute_hessian_landscape(
    eigenvals: List[float],
    angle_deg: float,
    lr: float,
    momentum_beta: float,
    start_point: List[float],
    steps: int = 35
) -> Dict[str, Any]:
    """Compute Hessian condition number kappa and compare SGD vs Momentum trajectories.

    Quadratic loss: L(w) = 1/2 w^T H w.
    Condition number kappa = lambda_max / lambda_min.
    Demonstrates ravines, cross-valley oscillation in vanilla SGD, and momentum dampening.
    """
    lam1 = max(0.1, float(eigenvals[0]))
    lam2 = max(0.1, float(eigenvals[1]))
    lam_max = max(lam1, lam2)
    lam_min = min(lam1, lam2)
    kappa = lam_max / lam_min

    # Rotation matrix R(theta)
    theta = np.radians(float(angle_deg))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    D = np.diag([lam1, lam2])
    H = R @ D @ R.T

    # Critical learning rate threshold for SGD stability: eta < 2 / lambda_max
    lr_max_stable = 2.0 / lam_max

    # Loss function L(w) = 1/2 w^T H w
    def loss_fn(w):
        return 0.5 * float(w @ H @ w)

    # Gradient grad L(w) = H w
    def grad_fn(w):
        return H @ w

    w0 = np.array(start_point, dtype=float)

    # 1. Vanilla SGD: w_{t+1} = w_t - lr * grad
    traj_sgd = [w0.tolist()]
    losses_sgd = [round(loss_fn(w0), 4)]
    w_sgd = w0.copy()
    for _ in range(steps):
        g = grad_fn(w_sgd)
        w_sgd = w_sgd - lr * g
        traj_sgd.append([round(float(w_sgd[0]), 4), round(float(w_sgd[1]), 4)])
        losses_sgd.append(round(loss_fn(w_sgd), 4))
        # Prevent numerical explosion
        if np.linalg.norm(w_sgd) > 50.0:
            break

    # 2. Momentum (Polyak): v_{t+1} = beta * v_t + grad, w_{t+1} = w_t - lr * v_{t+1}
    traj_mom = [w0.tolist()]
    losses_mom = [round(loss_fn(w0), 4)]
    w_mom = w0.copy()
    v_mom = np.zeros(2)
    for _ in range(steps):
        g = grad_fn(w_mom)
        v_mom = momentum_beta * v_mom + g
        w_mom = w_mom - lr * v_mom
        traj_mom.append([round(float(w_mom[0]), 4), round(float(w_mom[1]), 4)])
        losses_mom.append(round(loss_fn(w_mom), 4))
        if np.linalg.norm(w_mom) > 50.0:
            break

    # 3. Equipotential Contour Ellipses: 1/2 (lam1 * u^2 + lam2 * v^2) = C
    # Semi-axes in rotated coordinates: rx = sqrt(2 * C / lam1), ry = sqrt(2 * C / lam2)
    initial_loss = loss_fn(w0)
    base_levels = [0.25, 0.75, 1.5, 3.0, 6.0, 12.0, 24.0]
    contour_ellipses = []
    for c in base_levels:
        level_val = c * (max(initial_loss, 1.0) / 10.0)
        rx = float(np.sqrt(2.0 * level_val / lam1))
        ry = float(np.sqrt(2.0 * level_val / lam2))
        contour_ellipses.append({
            "level": round(level_val, 3),
            "rx": round(rx, 4),
            "ry": round(ry, 4),
            "angle_deg": round(float(angle_deg), 2)
        })

    # Oscillation metric: ratio of path length to net displacement
    def calc_oscillation(traj):
        if len(traj) < 2:
            return 1.0
        pts = np.array(traj)
        step_lens = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1))
        net_dist = np.linalg.norm(pts[-1] - pts[0])
        return round(float(step_lens / max(net_dist, 1e-4)), 2)

    return {
        "hessian_matrix": [[round(float(x), 4) for x in row] for row in H],
        "eigenvalues": [round(lam1, 3), round(lam2, 3)],
        "lambda_max": round(lam_max, 3),
        "lambda_min": round(lam_min, 3),
        "condition_number": round(kappa, 2),
        "angle_deg": round(float(angle_deg), 2),
        "lr": lr,
        "lr_max_stable": round(lr_max_stable, 4),
        "is_lr_unstable": bool(lr >= lr_max_stable),
        "momentum_beta": momentum_beta,
        "start_point": [round(float(w0[0]), 3), round(float(w0[1]), 3)],
        "trajectory_sgd": traj_sgd,
        "losses_sgd": losses_sgd,
        "trajectory_momentum": traj_mom,
        "losses_momentum": losses_mom,
        "final_loss_sgd": losses_sgd[-1] if losses_sgd else 0.0,
        "final_loss_momentum": losses_mom[-1] if losses_mom else 0.0,
        "oscillation_sgd": calc_oscillation(traj_sgd),
        "oscillation_momentum": calc_oscillation(traj_mom),
        "contour_ellipses": contour_ellipses
    }

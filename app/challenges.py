"""Interactive challenges and auto-grading engine for LA_ML.

Provides generator and verification algorithms for student practice tasks:
1. Matrix Transformation Matching (mapping vector u to v)
2. Space Un-collapsing / Inversion (finding W^(-1))
3. PCA Variance Maximization (finding principal component angle)
4. Least Squares Parameter Estimation
"""

import math
import random
from typing import Any, Dict, List
import numpy as np


CHALLENGE_TYPES = [
    {
        "id": "vector_mapping",
        "title": "Задача 1: Подбор матрицы перехода (Векторный таргетинг)",
        "difficulty": "Базовый",
        "description": "Подберите матрицу W (2x2), которая переводит вектор u в целевой вектор v."
    },
    {
        "id": "matrix_inversion",
        "title": "Задача 2: Обращение деформации (Инверсия пространства)",
        "difficulty": "Средний",
        "description": "Пространство было деформировано матрицей W. Найдите обратную матрицу W⁻¹, возвращающую систему координат к исходному единичному базису I."
    },
    {
        "id": "pca_variance",
        "title": "Задача 3: Охота за главной компонентой (PCA)",
        "difficulty": "Продвинутый",
        "description": "Поверните проекционную ось так, чтобы максимизировать дисперсию проекций точек и минимизировать потерю информации."
    },
    {
        "id": "svd_energy",
        "title": "Задача 4: Энергия сингулярного разложения (SVD)",
        "difficulty": "Средний",
        "description": "Определите минимальный ранг k низкоранговой аппроксимации, необходимый для сохранения не менее 90% энергии матрицы."
    },
    {
        "id": "rag_cosine",
        "title": "Задача 5: Векторный таргетинг RAG (Косинусное сходство)",
        "difficulty": "Базовый",
        "description": "Поверните вектор поискового запроса так, чтобы достичь максимального косинусного сходства (cos >= 0.95) с целевым семантическим кластером."
    },
    {
        "id": "class_separation",
        "title": "Задача 6: Разделение классов (Гиперплоскость и SVM)",
        "difficulty": "Средний",
        "description": "Подберите веса w₁, w₂ и порог b так, чтобы гиперплоскость wᵀx + b = 0 разделила красные и синие точки с точностью 100%."
    },
    {
        "id": "invariant_eigenvector",
        "title": "Задача 7: Поиск инвариантного направления (Собственный вектор)",
        "difficulty": "Продвинутый",
        "description": "Для оператора W поверните вектор v так, чтобы W v был строго коллинеарен v (W v = λ v)."
    },
    {
        "id": "least_squares_fit",
        "title": "Задача 8: Подбор проекции МНК на глаз",
        "difficulty": "Базовый",
        "description": "По визуальным точкам подберите наклон k и сдвиг b прямой, минимизируя длину вектора невязок ||e||."
    }
]


def generate_challenge(challenge_id: str) -> Dict[str, Any]:
    """Generate a challenge instance with randomized target parameters."""
    if challenge_id == "vector_mapping":
        # Generate simple integers for clear intuition
        u = [1.0, 1.0]
        # Target matrix with small integer coordinates
        while True:
            w11 = random.choice([-2, -1, 0, 1, 2])
            w12 = random.choice([-1, 0, 1, 2])
            w21 = random.choice([-1, 0, 1])
            w22 = random.choice([-2, -1, 1, 2])
            # Ensure not identity
            if w11 == 1 and w12 == 0 and w21 == 0 and w22 == 1:
                continue
            v = [w11 * u[0] + w12 * u[1], w21 * u[0] + w22 * u[1]]
            # Ensure target vector is non-zero (non-degenerate target)
            if abs(v[0]) + abs(v[1]) >= 1.0:
                break
        return {
            "id": "vector_mapping",
            "title": "Подбор матрицы перехода",
            "instruction": f"Подберите матрицу W, переводящую u = [{u[0]}, {u[1]}] в v = [{v[0]}, {v[1]}].",
            "u": u,
            "target_v": v,
            "hint": f"Вектор результата W u = [w11*{u[0]} + w12*{u[1]}, w21*{u[0]} + w22*{u[1]}]. Подберите такие w_ij, чтобы получить [{v[0]}, {v[1]}]."
        }

    elif challenge_id == "matrix_inversion":
        # Generate an invertible 2x2 matrix with integer determinant +-1 or +-2
        matrices = [
            [[1, 1], [0, 1]],  # shear
            [[2, 0], [0, 1]],  # scale
            [[0, -1], [1, 0]], # 90 deg rotation
            [[1, 2], [1, 3]],  # det = 1
            [[2, 1], [1, 1]],  # det = 1
            [[1, 0], [2, 1]],  # shear
        ]
        W = random.choice(matrices)
        W_np = np.array(W, dtype=float)
        W_inv = np.linalg.inv(W_np).tolist()

        return {
            "id": "matrix_inversion",
            "title": "Обращение деформации",
            "instruction": "Дана матрица деформации W. Найдите матрицу W_inv, чтобы W_inv * W = I.",
            "W": W,
            "hint": "Формула для матрицы 2x2: inv(A) = 1/det(A) * [[d, -b], [-c, a]].",
            "target_inverse": [[round(x, 3) for x in row] for row in W_inv]
        }

    elif challenge_id == "pca_variance":
        # Generate random rotated 2D Gaussian points
        angle = random.uniform(20.0, 70.0)  # degrees
        rad = math.radians(angle)
        R = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
        # Points stretched along X axis
        n = 30
        x_raw = np.random.normal(0, 2.5, n)
        y_raw = np.random.normal(0, 0.6, n)
        raw_pts = np.column_stack([x_raw, y_raw])
        rotated_pts = raw_pts @ R.T

        # Compute true PCA
        cov = np.cov(rotated_pts, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        true_v1 = eigvecs[:, idx[0]]
        true_angle = math.degrees(math.atan2(true_v1[1], true_v1[0])) % 180

        return {
            "id": "pca_variance",
            "title": "Охота за главной осью дисперсии",
            "instruction": "Задайте угол theta (в градусах) оси проекции, чтобы максимизировать сохраняемую дисперсию выборки.",
            "points": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in rotated_pts],
            "true_angle_deg": round(true_angle, 1),
            "hint": "Главная компонента направлена вдоль направления наибольшей вытянутости облака точек."
        }

    elif challenge_id == "svd_energy":
        options = [
            {"sigmas": [8.0, 3.0, 1.0], "threshold": 90.0},
            {"sigmas": [12.0, 2.0, 0.8], "threshold": 90.0},
            {"sigmas": [5.0, 4.0, 1.5], "threshold": 90.0},
            {"sigmas": [9.0, 4.0, 2.0], "threshold": 90.0},
        ]
        choice = random.choice(options)
        s = choice["sigmas"]
        total_energy = sum(x**2 for x in s)
        cum_energies = []
        cur = 0.0
        correct_rank = len(s)
        for i, val in enumerate(s):
            cur += val**2
            pct = (cur / total_energy) * 100
            cum_energies.append(round(pct, 1))
            if pct >= 90.0 and correct_rank == len(s):
                correct_rank = i + 1

        return {
            "id": "svd_energy",
            "title": "Энергия сингулярного разложения (SVD)",
            "instruction": f"Даны сингулярные числа матрицы: σ = {s}. Определите минимальный ранг k, чтобы сохранить не менее 90% энергии матрицы (сумма квадратов σᵢ).",
            "singular_values": s,
            "total_energy": round(total_energy, 2),
            "correct_rank": correct_rank,
            "cumulative_energies": cum_energies,
            "hint": "Формула энергии: E(k) = (σ₁² + ... + σₖ²) / (Σ σᵢ²). Рассчитайте сумму квадратов для k=1, 2, 3..."
        }

    elif challenge_id == "rag_cosine":
        target_angle = random.randint(20, 160)
        target_rad = math.radians(target_angle)
        target_vec = [round(math.cos(target_rad) * 2.5, 2), round(math.sin(target_rad) * 2.5, 2)]

        return {
            "id": "rag_cosine",
            "title": "Векторный таргетинг RAG (Косинусное сходство)",
            "instruction": f"Направьте поисковый запрос (угол θ) так, чтобы косинусное сходство с целевым документом d = {target_vec} составило не менее 0.95 (угол между векторами < 18°).",
            "target_vector": target_vec,
            "target_angle_deg": target_angle,
            "hint": "Косинусное сходство максимально (равно 1.0), когда вектор запроса сонаправлен с целевым документом."
        }

    elif challenge_id == "class_separation":
        # Generate 2 separable clusters (blue points +1, red points -1)
        angle = random.uniform(20.0, 70.0)
        rad = math.radians(angle)
        nx, ny = math.cos(rad), math.sin(rad)
        sep_dist = random.uniform(1.8, 2.5)

        points = []
        # Class +1
        c1_x, c1_y = nx * sep_dist, ny * sep_dist
        for _ in range(8):
            px = c1_x + random.uniform(-0.9, 0.9)
            py = c1_y + random.uniform(-0.9, 0.9)
            points.append({"x": round(px, 2), "y": round(py, 2), "label": 1})

        # Class -1
        c2_x, c2_y = -nx * sep_dist, -ny * sep_dist
        for _ in range(8):
            px = c2_x + random.uniform(-0.9, 0.9)
            py = c2_y + random.uniform(-0.9, 0.9)
            points.append({"x": round(px, 2), "y": round(py, 2), "label": -1})

        return {
            "id": "class_separation",
            "title": "Разделение классов (Гиперплоскость)",
            "instruction": "Подберите веса w1, w2 и сдвиг b, чтобы гиперплоскость w1*x + w2*y + b = 0 разделила синие (+1) и красные (-1) точки (Accuracy = 100%).",
            "points": points,
            "hint": "Вектор весов w = [w1, w2] задает нормаль, направленную в сторону синих точек (+1), а b сдвигает разделяющую прямую."
        }

    elif challenge_id == "invariant_eigenvector":
        cand_matrices = [
            [[2.0, 1.0], [1.0, 2.0]],     # Eig 3 (45 deg) & 1 (135 deg)
            [[3.0, 0.0], [1.0, 1.0]],     # Eig 3 (63.4 deg) & 1 (90 deg)
            [[1.0, 2.0], [0.0, 3.0]],     # Eig 1 (0 deg) & 3 (45 deg)
            [[2.0, -1.0], [-1.0, 2.0]],   # Eig 1 (45 deg) & 3 (135 deg)
            [[2.0, 0.0], [0.0, -1.0]],    # Eig 2 (0 deg) & -1 (90 deg)
            [[2.0, 1.0], [0.0, 1.0]],     # Eig 2 (0 deg) & 1 (135 deg)
        ]
        W = random.choice(cand_matrices)
        W_np = np.array(W, dtype=float)
        vals, vecs = np.linalg.eig(W_np)

        angles = []
        for i in range(2):
            v = vecs[:, i]
            ang = math.degrees(math.atan2(float(v[1].real), float(v[0].real))) % 180
            angles.append(round(ang, 1))

        return {
            "id": "invariant_eigenvector",
            "title": "Поиск инвариантного направления (Собственный вектор)",
            "instruction": f"Дана матрица оператора W = {W}. Поверните вектор v (угол θ), чтобы W v был строго коллинеарен v (W v = λ v).",
            "W": W,
            "target_eigenvalues": [round(float(v.real), 2) for v in vals],
            "target_angles_deg": angles,
            "hint": "При совпадении с собственным вектором стрелка W v лежит строго на одной прямой с вектором v."
        }

    elif challenge_id == "least_squares_fit":
        k_true = random.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
        b_true = random.choice([-2.0, -1.0, 0.0, 1.0, 2.0])
        xs = np.linspace(-3.0, 3.0, 8)
        points = []
        for x in xs:
            y = k_true * x + b_true + random.uniform(-0.5, 0.5)
            points.append([round(float(x), 2), round(float(y), 2)])

        pts_arr = np.array(points)
        X = np.column_stack([pts_arr[:, 0], np.ones(len(pts_arr))])
        y = pts_arr[:, 1]
        w_ols = np.linalg.pinv(X) @ y
        residuals_ols = y - X @ w_ols
        optimal_norm = float(np.linalg.norm(residuals_ols))

        return {
            "id": "least_squares_fit",
            "title": "Подбор проекции МНК на глаз",
            "instruction": "Подберите наклон k и сдвиг b прямой y = k*x + b, минимизируя длину вектора вертикальных невязок ||e||.",
            "points": points,
            "optimal_slope": round(float(w_ols[0]), 2),
            "optimal_intercept": round(float(w_ols[1]), 2),
            "optimal_residual_norm": round(optimal_norm, 3),
            "hint": "Выставьте прямую так, чтобы сумма квадратов вертикальных отклонений точек от прямой была минимальной."
        }

    else:
        raise ValueError(f"Unknown challenge id: {challenge_id}")


def verify_challenge(challenge_id: str, submission: Dict[str, Any]) -> Dict[str, Any]:
    """Verify student submission and return feedback with score."""
    if challenge_id == "vector_mapping":
        u = np.array(submission.get("u", [1.0, 1.0]), dtype=float)
        target_v = np.array(submission.get("target_v", [0.0, 0.0]), dtype=float)
        W_user = np.array(submission.get("matrix", [[1, 0], [0, 1]]), dtype=float)

        res_v = W_user @ u
        error = float(np.linalg.norm(res_v - target_v))
        success = error < 0.15

        return {
            "success": success,
            "error": round(error, 4),
            "user_result": [round(float(x), 3) for x in res_v],
            "target": target_v.tolist(),
            "message": "Отлично! Вектор точно переведен в цель!" if success else f"Погрешность: {round(error, 3)}. Получился вектор [{round(float(res_v[0]), 2)}, {round(float(res_v[1]), 2)}]."
        }

    elif challenge_id == "matrix_inversion":
        W = np.array(submission.get("W", [[1, 0], [0, 1]]), dtype=float)
        W_user = np.array(submission.get("matrix", [[1, 0], [0, 1]]), dtype=float)

        product = W_user @ W
        identity = np.eye(2)
        error = float(np.linalg.norm(product - identity, ord='fro'))
        success = error < 0.2

        return {
            "success": success,
            "error": round(error, 4),
            "product": [[round(float(x), 3) for x in row] for row in product],
            "message": "Верно! Произведение матриц дает единичную матрицу I." if success else f"Матрица не является обратной. Ошибка нормы: {round(error, 3)}."
        }

    elif challenge_id == "pca_variance":
        user_angle = float(submission.get("angle_deg", 0.0)) % 180
        true_angle = float(submission.get("true_angle_deg", 0.0)) % 180

        diff = abs(user_angle - true_angle)
        # Accounting for 180-degree symmetry of axes
        angle_err = min(diff, 180.0 - diff)
        success = angle_err <= 12.0  # within 12 degrees tolerance

        return {
            "success": success,
            "angle_error_deg": round(angle_err, 2),
            "user_angle": round(user_angle, 1),
            "true_angle": round(true_angle, 1),
            "message": f"Блестяще! Вы нашли направление максимальной дисперсии с точностью до {round(angle_err, 1)}°." if success else f"Отклонение от истинной главной оси составляет {round(angle_err, 1)}°. Попробуйте повернуть ось ближе к эллипсу рассеяния."
        }

    elif challenge_id == "svd_energy":
        user_rank = int(submission.get("rank", 1))
        correct_rank = int(submission.get("correct_rank", 1))
        success = (user_rank == correct_rank)
        cum_e = submission.get("cumulative_energies", [])
        pct = cum_e[user_rank - 1] if 0 < user_rank <= len(cum_e) else 0.0

        return {
            "success": success,
            "user_rank": user_rank,
            "correct_rank": correct_rank,
            "energy_pct": pct,
            "message": f"Верно! При k={correct_rank} сохраняется {pct}% энергии (>= 90%)." if success else f"Не совсем. При k={user_rank} сохраняется {pct}% энергии. Правильный минимальный ранг: k={correct_rank}."
        }

    elif challenge_id == "rag_cosine":
        user_angle = float(submission.get("angle_deg", 0.0))
        target_angle = float(submission.get("target_angle_deg", 0.0))
        diff_deg = abs((user_angle - target_angle + 180.0) % 360.0 - 180.0)
        diff_rad = math.radians(diff_deg)
        cos_sim = math.cos(diff_rad)
        success = (cos_sim >= 0.95)

        return {
            "success": success,
            "cosine_similarity": round(cos_sim, 4),
            "user_angle": round(user_angle, 1),
            "target_angle": round(target_angle, 1),
            "message": f"Отличный таргетинг! Косинусное сходство: {round(cos_sim, 4)} >= 0.95. Документ успешно извлечен (Top-1 RAG retrieval)!" if success else f"Косинусное сходство: {round(cos_sim, 4)} < 0.95 (угол отклонения: {round(diff_deg, 1)}°). Поверните запрос ближе к вектору цели."
        }

    elif challenge_id == "class_separation":
        points = submission.get("points", [])
        if not points:
            return {"success": False, "message": "Список точек пуст."}
        w1 = float(submission.get("w1", 1.0))
        w2 = float(submission.get("w2", 0.0))
        b = float(submission.get("b", 0.0))

        w = np.array([w1, w2], dtype=float)
        norm_w = float(np.linalg.norm(w))
        if norm_w < 1e-6:
            return {"success": False, "message": "Вектор весов w не должен быть нулевым!"}

        correct = 0
        min_margin = 999.0
        for pt in points:
            x = np.array([pt["x"], pt["y"]], dtype=float)
            label = 1 if pt.get("label", 1) >= 0 else -1
            score = float(w @ x + b)
            if (score >= 0 and label == 1) or (score < 0 and label == -1):
                correct += 1
            dist = (label * score) / norm_w
            if dist < min_margin:
                min_margin = dist

        total = len(points)
        acc = (correct / total * 100.0) if total > 0 else 0.0
        success = (correct == total)

        return {
            "success": success,
            "accuracy": round(acc, 1),
            "correct_count": correct,
            "total_count": total,
            "margin": round(max(0.0, min_margin), 3),
            "message": f"Блестяще! Все {total} точек разделены со 100% точностью! Полоса отступа Margin = {round(max(0.0, min_margin), 3)}." if success else f"Точность: {round(acc, 1)}% ({correct}/{total}). Разделение не полное (требуется 100%). Подстройте наклон весов w или сдвиг b."
        }

    elif challenge_id == "invariant_eigenvector":
        W = np.array(submission.get("W", [[1, 0], [0, 1]]), dtype=float)
        angle_deg = float(submission.get("angle_deg", 0.0))
        rad = math.radians(angle_deg)
        v = np.array([math.cos(rad), math.sin(rad)])
        Wv = W @ v

        norm_Wv = float(np.linalg.norm(Wv))
        if norm_Wv < 1e-6:
            success = True
            lam = 0.0
            sin_diff = 0.0
        else:
            # Cross product check for collinearity: |v_x * Wv_y - v_y * Wv_x| / ||Wv||
            sin_diff = abs(v[0] * Wv[1] - v[1] * Wv[0]) / norm_Wv
            success = (sin_diff < 0.08)  # < 4.5 degrees
            lam = round(float(np.dot(v, Wv)), 3)

        return {
            "success": success,
            "user_angle_deg": round(angle_deg, 1),
            "eigenvalue": lam if success else None,
            "collinearity_error": round(float(sin_diff), 4),
            "message": f"Великолепно! Вектор v инвариантен: W v = {lam} v (Собственное число λ = {lam})!" if success else f"Вектор W v не коллинеарен v (ошибка коллинеарности: {round(sin_diff, 3)}). Поверните вектор ближе к инвариантной оси."
        }

    elif challenge_id == "least_squares_fit":
        points = submission.get("points", [])
        if not points:
            return {"success": False, "message": "Список точек пуст."}
        slope = float(submission.get("slope", 0.0))
        intercept = float(submission.get("intercept", 0.0))
        opt_norm = float(submission.get("optimal_residual_norm", 1.0))

        residuals_sq = 0.0
        for p in points:
            y_hat = slope * p[0] + intercept
            residuals_sq += (p[1] - y_hat) ** 2
        user_norm = math.sqrt(residuals_sq)

        ratio = (user_norm / opt_norm) if opt_norm > 1e-6 else 1.0
        success = (ratio <= 1.25)  # within 25% of theoretical optimal norm

        return {
            "success": success,
            "user_residual_norm": round(user_norm, 3),
            "optimal_residual_norm": round(opt_norm, 3),
            "ratio": round(ratio, 2),
            "message": f"Отличный глазомер! Ваша норма невязок ||e|| = {round(user_norm, 3)} практически совпадает с теоретическим минимумом МНК {round(opt_norm, 3)} (отклонение {abs(round((ratio - 1)*100, 1))}%)!" if success else f"Норма невязок ||e|| = {round(user_norm, 3)} превышает оптимум МНК {round(opt_norm, 3)} на {round((ratio - 1)*100, 1)}%. Подкорректируйте прямую."
        }

    return {"success": False, "message": "Неизвестный тип задачи"}

"""
Тесты для модуля clusters
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sklearn.datasets import make_blobs
from clusters import cluster_data


def test_kmeans_basic():
    """Тест базового KMeans"""
    print("  Тест 1: Базовый KMeans")
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=42)
    model = cluster_data(X, 'kmeans', n_clusters=3, choose_optimal=False)

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'labels_'), "Модель должна иметь метки"
    assert len(np.unique(model.labels_)) == 3, f"Неверное количество кластеров: {len(np.unique(model.labels_))}"
    print("    ✅ KMeans работает")


def test_kmeans_optimal():
    """Тест KMeans с выбором оптимального числа кластеров"""
    print("  Тест 2: Оптимальный KMeans")
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=42)
    model = cluster_data(X, 'kmeans', choose_optimal=True, max_clusters=5)

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'labels_'), "Модель должна иметь метки"
    print("    ✅ Оптимальный KMeans работает")


def test_dbscan():
    """Тест DBSCAN"""
    print("  Тест 3: DBSCAN")
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.5, random_state=42)
    model = cluster_data(X, 'dbscan')

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'labels_'), "Модель должна иметь метки"
    print("    ✅ DBSCAN работает")


def test_agglomerative():
    """Тест Agglomerative Clustering"""
    print("  Тест 4: Agglomerative Clustering")
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=42)
    model = cluster_data(X, 'agglomerative', n_clusters=3, choose_optimal=False)

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'labels_'), "Модель должна иметь метки"
    assert len(np.unique(model.labels_)) == 3, f"Неверное количество кластеров: {len(np.unique(model.labels_))}"
    print("    ✅ Agglomerative работает")


def test_gmm():
    """Тест Gaussian Mixture Model"""
    print("  Тест 5: Gaussian Mixture Model")
    X, _ = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, random_state=42)
    model = cluster_data(X, 'gmm', n_clusters=3, choose_optimal=False)

    assert model is not None, "Модель не создана"
    assert hasattr(model, 'predict'), "Модель должна иметь метод predict"
    print("    ✅ GMM работает")


def run_tests():
    """Запуск всех тестов кластеризации"""
    print("\n" + "=" * 40)
    print("Запуск тестов кластеризации")
    print("=" * 40)

    tests = [
        test_kmeans_basic,
        test_kmeans_optimal,
        test_dbscan,
        test_agglomerative,
        test_gmm
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"    ❌ Ошибка: {e}")
        except Exception as e:
            failed += 1
            print(f"    ❌ Неожиданная ошибка: {e}")

    print(f"\nРезультат: {passed} прошло, {failed} упало")
    return passed, failed


if __name__ == "__main__":
    run_tests()
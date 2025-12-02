from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def cluster_data(x, algorithm: str = 'kmeans', /, n_clusters: int = 3,
                 choose_optimal: bool = False, max_clusters: int = 10,
                 random_state: int = -1):
    """
    Универсальный метод кластеризации с поддержкой разных алгоритмов.

    Параметры:
    -----------
    x : array-like
        Данные для кластеризации
    algorithm : str, default='kmeans'
        Алгоритм кластеризации ('kmeans', 'dbscan', 'agglomerative', 'gmm')
    n_clusters : int, default=3
        Количество кластеров (для некоторых алгоритмов)
    choose_optimal : bool, default=False
        Автоматический выбор оптимальных параметров
    max_clusters : int, default=10
        Максимальное количество кластеров для перебора
    random_state : int, default=None
        Seed для воспроизводимости
    return_scores : bool, default=False
        Если True, возвращает scores качества кластеризации

    Возвращает:
    -----------
    model : clustering model
        Обученная модель кластеризации
    labels : array
        Метки кластеров
    scores : dict (optional)
        Метрики качества кластеризации

    Алгоритмы распределения:
    ====================
    Название________Скорость_____Работа с сшумом___________Фитчи

    kmeans__________Высокая________Плохая___________Требует указания числа кластеров\n
    ________________________________________________Работает только с spherical кластерами\n
    ________________________________________________Чувствителен к масштабу данных\n
    \n
    dbscan__________Средняя_______Отличная__________Автоматически определяет число кластеров\n
    ________________________________________________Находит кластеры произвольной формы\n
    ________________________________________________Выделяет выбросы как шум(-1)\n
    \n
    agglomerative____Низкая________Плохая___________Иерархическая структура кластеров\n
    ________________________________________________Визуализация через дендрограмму\n
    ________________________________________________Чувствителен к шуму и выбросам \n
    \n
    gmm______________Низкая_______Средняя___________Вероятностная модель кластеризации\n
    ________________________________________________Мягкое присвоение кластеров(вероятности)\n
    ________________________________________________Может находить elliptical кластеры\n
    """

    # Normalising data
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    scores = {}

    if algorithm == 'kmeans':
        from sklearn.cluster import KMeans

        if choose_optimal:
            model, labels = kmeans_clustering(
                x_scaled,
                choose_optimal=True,
                max_clusters=max_clusters,
                random_state=random_state if random_state != -1 else None
            )
        else:
            model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state if random_state != -1 else None,
                n_init=10
            )
            labels = model.fit_predict(x_scaled)

    elif algorithm == 'dbscan':
        from sklearn.cluster import DBSCAN

        # DBSCAN auto num detective
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(x_scaled)
        n_clusters_found = len(np.unique(labels[labels != -1]))  # Ignoring  (-1)
        print(f"DBSCAN found {n_clusters_found} clusters (+ noise)")

    elif algorithm == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering

        if choose_optimal:
            best_score = -1
            best_n = 2
            best_model = None
            best_labels = None

            for n in range(2, max_clusters + 1):
                model = AgglomerativeClustering(n_clusters=n)
                labels = model.fit_predict(x_scaled)

                if len(np.unique(labels)) > 1:
                    score = silhouette_score(x_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
                        best_model = model
                        best_labels = labels

            print(f"Optimal clusters for Agglomerative: {best_n}")
            model, labels = best_model, best_labels
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(x_scaled)

    elif algorithm == 'gmm':
        from sklearn.mixture import GaussianMixture

        if choose_optimal:
            best_score = -1
            best_n = 2
            best_model = None
            best_labels = None

            for n in range(2, max_clusters + 1):
                model = GaussianMixture(
                    n_components=n,
                    random_state=random_state if random_state != -1 else None
                )
                labels = model.fit_predict(x_scaled)

                if len(np.unique(labels)) > 1:
                    score = silhouette_score(x_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
                        best_model = model
                        best_labels = labels

            print(f"Optimal components for GMM: {best_n}")
            model, labels = best_model, best_labels
        else:
            model = GaussianMixture(
                n_components=n_clusters,
                random_state=random_state if random_state != -1 else None
            )
            labels = model.fit_predict(x_scaled)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: 'kmeans', 'dbscan', 'agglomerative', 'gmm'")

    # Testing model
    if len(np.unique(labels)) > 1 and -1 not in labels:  # Checking clusters
        scores['silhouette'] = silhouette_score(x_scaled, labels)
        scores['calinski_harabasz'] = calinski_harabasz_score(x_scaled, labels)
        scores['davies_bouldin'] = davies_bouldin_score(x_scaled, labels)

        print(f"Clustering quality metrics:")
        print(f"  Silhouette Score: {scores['silhouette']:.4f}")
        print(f"  Calinski-Harabasz: {scores['calinski_harabasz']:.4f}")
        print(f"  Davies-Bouldin: {scores['davies_bouldin']:.4f}")

    print(f"Number of clusters found: {len(np.unique(labels))}")

    return model



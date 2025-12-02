from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def t_kneighbors(X, y,/, test_size:float = 0.1, choose_optimal: bool = False, neighbors: int = 1,
               random_state: int = -1):
    """
    Implementation of the k-nearest neighbors (KNN) method for classification.

    Parameters:
    ================================
    x : array-like of shape (n_samples, n_features)
        Array of features for model training
    y : array-like of shape (n_samples,)
        Target values (class labels)
    test_size : float, default=0.1
        Percentage of test data (from 0.0 to 1.0)
    choose_optimal : bool, default=False
        If True, it automatically selects the optimal number of neighbors (1-10)
    neighbors : int, default=1
        Number of neighbors to use (only if choose_optimal=False)

    Returns:
    ================================
    model : KNeighborsClassifier
        The trained k-nearest neighbors model

    Using:
    ================================
    >>> import FastAI as ai
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = ai.kneighbors(X, y, test_size=0.2, choose_optimal=True)
    """

    #Split data for train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state if random_state != -1 else None)

    if not choose_optimal:
        # Create model
        model = KNeighborsClassifier(n_neighbors=neighbors)
        model.fit(x_train, y_train)

        # Model testing
        if test_size > 0:
            print(f"Model currency\n "
                     f"on test data: {model.score(x_test, y_test)}\n "
                     f"on train data: {model.score(x_train, y_train)}")

        return model

    #Choosing better neighbors count, and return that model
    else:
        #The test data is needed to compare the results of the work
        if test_size == 0:
            raise ValueError("Warring: Can't choose the best model without TEST DATA")
            return

        results = []
        #We teach 10 models and choose better one (if you need more than 10, change range() values)
        for i in range(1,11):
            model = KNeighborsClassifier(n_neighbors=neighbors)
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)
            results.append(accuracy)

            if accuracy == 1:
                break

        print(f"The better model currency is {max(results)} with {results.index(max(results))+1} neighbors")

        model = KNeighborsClassifier(n_neighbors=results.index(max(results))+1)
        model.fit(x_train, y_train)

        return model

def kneighbors(neighbors: int = 1):
    """Return: Not taught model"""
    return KNeighborsClassifier(n_neighbors=neighbors)



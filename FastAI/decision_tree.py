from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def t_decision_tree(x, y, /, test_size: float = 0.1, choose_optimal: bool = False,
                  max_depth: int = None, random_state: int = -1):
    """
    Implementation of the decision tree method for classification.

    Parameters:
    ===================
    x : array-like of shape (n_samples, n_features)
        Array of features for model training
    y : array-like of shape (n_samples,)
        Target values (class labels)
    test_size : float, default=0.1
        Percentage of test data (from 0.0 to 1.0)
    choose_optimal : bool, default=False
        If True, it automatically selects the optimal depth of the tree.
    max_depth : int, default=None
        Maximum depth of the tree (only if choose_optimal=False)
    random_state : int, default=None
        Seed for reproducibility of results

    Returns:
    ================
    model : DecisionTreeClassifier
        Trained decision tree model

    Examples:
    =================
    >>> import FastAI as ai
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> model = ai.decision_tree(X, y, test_size=0.2, choose_optimal=True)
    """

    #Split data for train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state if random_state != -1 else None, stratify=y
    )

    if not choose_optimal:
        # Create and teach
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(x_train, y_train)

        # Model testing
        if test_size > 0:
            print(f"Model currency\n "
                     f"on test data: {model.score(x_test, y_test)}\n "
                     f"on train data: {model.score(x_train, y_train)}")

        return model

    else:
        if test_size == 0:
            raise ValueError("Warning: Cannot choose the best model without TEST DATA")

        best_accuracy = 0
        best_depth = 1
        results = []

        # Test any tree depth from 1 to 10
        for depth in range(1, 11):
            model = DecisionTreeClassifier(
                max_depth=depth,
                random_state=random_state
            )
            model.fit(x_train, y_train)

            accuracy = model.score(x_test, y_test)
            results.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_depth = depth

        print(f"Best test accuracy: {best_accuracy:.4f} with max_depth={best_depth}")
        print(f"All results: {dict(zip(range(1, 11), [f'{acc:.4f}' for acc in results]))}")

        # teach the best model
        best_model = DecisionTreeClassifier(
            max_depth=best_depth,
            random_state=random_state if random_state != -1 else None
        )
        best_model.fit(x_train, y_train)

        return best_model

def decision_tree(max_depth: int = 1, random_state: int = None):
    """Return: Not taught model"""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state if random_state != -1 else None
    )


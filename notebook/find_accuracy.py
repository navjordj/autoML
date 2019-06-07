
def evaluate(y_pred, y_test):
    return {'accuracy': str(accuracy_score(y_test, y_pred)),
            'precision': str(precision_score(y_test, y_pred)),
            'recall': str(recall_score(y_test, y_pred)),
            'f1': str(f1_score(y_test, y_pred))}


if __name__ == '__main__':
    print(evaluate([0, 1, 0], [1, 1, 0]))

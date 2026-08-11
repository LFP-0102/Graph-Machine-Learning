def accuracy(pred, label):
    return (pred == label).float().mean()

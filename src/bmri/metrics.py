import numpy as np

def evaluate_segmentation(a, b, threshold=0.5):
    """
    Calculates the Dice Similarity Coefficient (DSC), Positive Predictive Value (PPV),
    Sensitivity, and Specificity between two numpy arrays.

    Args:
    a, b (numpy arrays): Two arrays of equal size.
    threshold (float, optional): Threshold value for binarizing the input arrays. Default is 0.5.

    Returns:
    A dictionary containing the DSC, PPV, Sensitivity, and Specificity between the two arrays.
    """
    a = np.array(a)
    b = np.array(b)
    
    assert a.shape == b.shape, "Arrays must have the same shape."
    

    # Apply binary thresholding to the input arrays
    a = (a > threshold).astype(int)
    b = (b > threshold).astype(int)

    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN), and True Negatives (TN)
    TP = np.logical_and(a, b).sum()
    FP = np.logical_and(a, np.logical_not(b)).sum()
    FN = np.logical_and(np.logical_not(a), b).sum()
    TN = np.logical_and(np.logical_not(a), np.logical_not(b)).sum()

    # Calculate Dice Similarity Coefficient (DSC)
    DSC = (2. * TP) / (a.sum() + b.sum())

    # Calculate Positive Predictive Value (PPV)
    PPV = TP / (TP + FP)

    # Calculate Sensitivity (TPR)
    Sensitivity = TP / (TP + FN)

    # Calculate Specificity (TNR)
    Specificity = TN / (FP + TN)

    # Return the results as a dictionary
    return {'DSC': DSC, 'PPV': PPV, 'Sensitivity': Sensitivity, 'Specificity': Specificity}

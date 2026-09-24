import numpy as np

def detect_ood(predictions, threshold=0.6):
    confidence = np.max(predictions)

    if confidence < threshold:
        return True, confidence  # Unknown
    return False, confidence
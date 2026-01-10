def unscale(mean, scale, arr):
    """
    Reverts standard scaling.

    Args:
        mean (float): Feature mean.
        scale (float): Feature scale.
        arr (np.array): Scaled values.

    Returns:
        np.array: Unscaled values.
    """
    return mean + arr * scale

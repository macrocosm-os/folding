from typing import List


def divide_decreasing(
    amount_to_distribute: float, number_of_elements: int
) -> List[float]:
    # Instead of going from n to 0, we go from n to 1
    # This gives us weights that look like [n, n-1, n-2, ..., 1]
    weights = [number_of_elements - i for i in range(number_of_elements)]

    # Calculate scaling factor to make weights sum to amount_to_distribute
    total_weight = sum(weights)
    scaling_factor = amount_to_distribute / total_weight

    # Scale all weights proportionally
    return [w * scaling_factor for w in weights]

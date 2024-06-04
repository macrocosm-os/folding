from typing import List


def divide_decreasing(
    amount_to_distribute: float, number_of_elements: int
) -> List[float]:
    # Calculate the fixed decrease amount d
    d = 2 * amount_to_distribute / ((number_of_elements - 1) * number_of_elements)

    # Calculate the first value a1
    a1 = (
        amount_to_distribute + d * (number_of_elements - 1) * number_of_elements / 2
    ) / number_of_elements

    # Calculate the n values
    values = [a1 - i * d for i in range(number_of_elements)]

    return values

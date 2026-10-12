"""Price calculation for the example shop."""


def calculate_total(unit_price, quantity, discount=0):
    if not 0 <= discount <= 1:
        raise ValueError("Discount must be between zero and one")
    return round(unit_price * quantity * (1 - discount), 2)

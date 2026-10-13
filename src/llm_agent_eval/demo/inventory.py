"""Inventory operations used by checkout and order cancellation."""


def reserve_stock(inventory, product_id, quantity):
    if quantity <= 0 or inventory.get(product_id, 0) < quantity:
        raise ValueError("Insufficient stock or invalid quantity")
    inventory[product_id] -= quantity


def restore_stock(inventory, product_id, quantity):
    """Restore inventory when an order is cancelled."""
    inventory[product_id] = inventory.get(product_id, 0) + quantity

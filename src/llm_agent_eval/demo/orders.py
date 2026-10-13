"""Small demonstration shop, deliberately containing an inventory cancellation bug."""
from .inventory import reserve_stock


def place_order(inventory, product_id, quantity):
    reserve_stock(inventory, product_id, quantity)
    return {"product_id": product_id, "quantity": quantity, "status": "placed"}


def cancel_order(order, inventory):
    """Cancel an order. Reserved inventory should become available again."""
    if order["status"] == "cancelled":
        return order
    order["status"] = "cancelled"
    return order

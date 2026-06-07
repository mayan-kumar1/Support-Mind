from logger import get_logger

logger = get_logger(__name__)

ORDERS_DB = {
    "ORD001": {
        "order_id": "ORD001",
        "customer": "Ravi Sharma",
        "item": "Wireless Headphones",
        "status": "shipped",
        "estimated_delivery": "2026-06-08",
        "tracking_number": "TRK998877",
    },
    "ORD002": {
        "order_id": "ORD002",
        "customer": "Priya Mehta",
        "item": "Running Shoes",
        "status": "out_for_delivery",
        "estimated_delivery": "2026-06-06",
        "tracking_number": "TRK112233",
    },
    "ORD003": {
        "order_id": "ORD003",
        "customer": "Arjun Patel",
        "item": "Laptop Stand",
        "status": "delivered",
        "estimated_delivery": "2026-06-01",
        "tracking_number": "TRK445566",
    },
    "ORD004": {
        "order_id": "ORD004",
        "customer": "Sneha Reddy",
        "item": "Yoga Mat",
        "status": "processing",
        "estimated_delivery": "2026-06-10",
        "tracking_number": None,
    },
}

RETURNS_DB = {}


def get_order_status(order_id: str) -> dict:
    logger.info("Fetching order status  order_id: %s", order_id)

    order = ORDERS_DB.get(order_id.upper())

    if not order:
        logger.warning("Order not found  order_id: %s", order_id)
        return {
            "success": False,
            "error": f"Order {order_id} not found. Please check your order ID and try again.",
        }

    logger.info("Order found  order_id: %s  status: %s", order_id, order["status"])
    return {
        "success": True,
        "order_id": order["order_id"],
        "item": order["item"],
        "status": order["status"],
        "estimated_delivery": order["estimated_delivery"],
        "tracking_number": order["tracking_number"],
    }


def initiate_return(order_id: str, reason: str = "Not specified") -> dict:
    logger.info("Initiating return  order_id: %s  reason: %s", order_id, reason)

    order = ORDERS_DB.get(order_id.upper())

    if not order:
        logger.warning("Return failed  order not found  order_id: %s", order_id)
        return {
            "success": False,
            "error": f"Order {order_id} not found. Please check your order ID.",
        }

    if order["status"] != "delivered":
        logger.warning(
            "Return failed  not delivered  order_id: %s  status: %s",
            order_id,
            order["status"],
        )
        return {
            "success": False,
            "error": f"Order {order_id} cannot be returned yet as it has not been delivered. Current status: {order['status']}.",
        }

    if order_id.upper() in RETURNS_DB:
        logger.warning("Return already exists  order_id: %s", order_id)
        return {
            "success": False,
            "error": f"A return has already been initiated for order {order_id}.",
        }

    return_id = f"RET{order_id.upper()}"
    RETURNS_DB[order_id.upper()] = {
        "return_id": return_id,
        "order_id": order_id.upper(),
        "item": order["item"],
        "reason": reason,
        "status": "initiated",
    }

    logger.info("Return initiated  return_id: %s", return_id)
    return {
        "success": True,
        "return_id": return_id,
        "item": order["item"],
        "status": "initiated",
        "next_steps": "A prepaid return label will be sent to your registered email within 24 hours. Refund will be processed within 3 to 5 business days of receiving the item.",
    }

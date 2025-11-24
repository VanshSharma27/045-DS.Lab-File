def process_sale(inventory, sku, qty_sold):
    """
    Takes three inputs:
     inventory → current stock list
     sku → the item to sell
     qty_sold → how many units are sold
    Decrease stock of specific SKU based on sales.
    If SKU not found, notify user.
    If stock is insufficient, notify user.
    Args:
        inventory (list of tuples): [(SKU, quantity), ...]
        sku (int): SKU identifier to process sale
        qty_sold (int): Quantity sold
    Returns:
        updated_inventory (list of tuples)
    """
    updated_inventory = []  # Creates a new empty list called updated_inventory.
    sku_found = False
    for item in inventory:
        current_sku, current_qty = item
        if current_sku == sku:
            sku_found = True
            if current_qty >= qty_sold:
                updated_inventory.append((current_sku, current_qty - qty_sold))  # Append the updated tuple (SKU, new_qty) to updated_inventory.
                print(f"Sale processed: {qty_sold} units of SKU {sku}.")
            else:
                updated_inventory.append((current_sku, current_qty))
                print(f"Insufficient stock for SKU {sku}. Available: {current_qty}")
        else:
            updated_inventory.append(item)
    if not sku_found:
        print(f"SKU {sku} not found in inventory.")
    return updated_inventory


def identify_zero_stock(inventory):
    """
    Identify all SKUs with zero stock.
    Args:
        inventory (list of tuples): [(SKU, quantity), ...]
    Returns:
        zero_stock_list (list of int): SKUs with zero quantity
    """
    zero_stock_list = [sku for sku, qty in inventory if qty == 0]
    if zero_stock_list:
        print(f"Zero stock SKUs: {zero_stock_list}")
    else:
        print("No zero stock items found.")
    return zero_stock_list


# --- Interactive Usage ---
if __name__ == "__main__":
    inventory = [(101, 50), (102, 20), (103, 0)]
    print("Initial Inventory:", inventory)

    while True:
        try:
            sku_input = input("Enter SKU to sell (or 'q' to quit): ")
            if sku_input.lower() == 'q':
                break
            sku = int(sku_input)
            qty_sold = int(input("Enter quantity sold: "))
            inventory = process_sale(inventory, sku, qty_sold)
        except ValueError:
            print("Invalid input. Please enter numeric values for SKU and quantity.")

    zero_stock_items = identify_zero_stock(inventory)
    print("Final Inventory:", inventory)

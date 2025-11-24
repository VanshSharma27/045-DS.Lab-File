back_stack = []
forward_stack = []
current_page = None

def visit_page(page):
    global current_page
    if current_page is not None:
        back_stack.append(current_page)
    current_page = page
    forward_stack.clear()
    print(f"Visited: {current_page}")

def go_back():
    global current_page
    if not back_stack:
        print("No previous pages.")
        return
    if current_page is not None:
        forward_stack.append(current_page)
    current_page = back_stack.pop()
    print(f"Back to: {current_page}")

def go_forward():
    global current_page
    if not forward_stack:
        print("No forward pages.")
        return
    if current_page is not None:
        back_stack.append(current_page)
    current_page = forward_stack.pop()
    print(f"Forward to: {current_page}")

def delete_page():
    global current_page
    page = input("Enter page name to delete from history: ").strip()
    if page in back_stack:
        back_stack.remove(page)
        print(f"Deleted {page} from back history.")
    elif page in forward_stack:
        forward_stack.remove(page)
        print(f"Deleted {page} from forward history.")
    elif page == current_page:
        print(f"Deleted current page: {current_page}")
        current_page = None
    else:
        print("Page not found in history.")

def show_history():
    print("\n--- Browser History ---")
    print("Back Stack :", " -> ".join(back_stack) if back_stack else "Empty")
    print("Current    :", current_page if current_page else "None")
    print("Forward    :", " -> ".join(forward_stack) if forward_stack else "Empty")
    print("-----------------------\n")

def main():
    while True:
        print("\n1. Visit Page")
        print("2. Back")
        print("3. Forward")
        print("4. Delete Page")
        print("5. Show History")
        print("6. Exit")
        choice = input("Enter choice: ").strip()
        if choice == "1":
            page = input("Enter page name (e.g. Google, GitHub): ").strip()
            visit_page(page)
        elif choice == "2":
            go_back()
        elif choice == "3":
            go_forward()
        elif choice == "4":
            delete_page()
        elif choice == "5":
            show_history()
        elif choice == "6":
            print("Exiting browser simulation.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()

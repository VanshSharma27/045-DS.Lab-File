def reverse_string(s):
    stack = []
    for char in s:
        stack.append(char)
    reversed_str = ""
    while stack:
        reversed_str += stack.pop()
    return reversed_str

def check_balanced(expr):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    
    for char in expr:
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    
   
    return len(stack) == 0

def undo_operations(operations):
    stack = []
    for op in operations:
        if op == "UNDO":
            if stack:
                stack.pop()
        else:
            stack.append(op)
    return "".join(stack)

while True:
    print("\n--- Stack Applications ---")
    print("1. Reverse a String")
    print("2. Check Balanced Parentheses")
    print("3. Undo Operations")
    print("4. Exit")
    
    choice = input("Enter your choice: ")
    
    if choice == "1":
        s = input("Enter a string: ")
        print("Reversed:", reverse_string(s))
    
    elif choice == "2":
        expr = input("Enter expression with brackets: ")
        if check_balanced(expr):
            print("Balanced")
        else:
            print("Not Balanced")
    
    elif choice == "3":
        n = int(input("How many operations? "))
        ops = []
        for i in range(n):
            op = input("Enter character or UNDO: ")
            ops.append(op)
        print("Final Text:", undo_operations(ops))
    
    elif choice == "4":
        print("Exiting...")
        break
    
    else:
        print("Invalid choice, try again.")
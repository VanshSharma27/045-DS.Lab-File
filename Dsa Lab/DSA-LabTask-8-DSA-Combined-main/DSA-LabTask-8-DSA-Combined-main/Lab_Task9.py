# ============================================================
# LAB 1 : CIRCULAR LINKED LIST (Insert, Delete Begin, Delete End)
# ============================================================

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class CircularLinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        new_node = Node(data)
        if self.head is None:
            new_node.next = new_node
            self.head = new_node
            return
        temp = self.head
        while temp.next != self.head:
            temp = temp.next
        temp.next = new_node
        new_node.next = self.head

    def delete_begin(self):
        if self.head is None:
            print("CLL Empty")
            return
        
        if self.head.next == self.head:
            self.head = None
            print("Deleted the only node")
            return
        
        last = self.head
        while last.next != self.head:
            last = last.next
        
        last.next = self.head.next
        self.head = self.head.next
        print("Deleted from beginning")

    def delete_end(self):
        if self.head is None:
            print("CLL Empty")
            return

        if self.head.next == self.head:
            self.head = None
            print("Deleted the only node")
            return

        prev = None
        temp = self.head
        while temp.next != self.head:
            prev = temp
            temp = temp.next

        prev.next = self.head
        print("Deleted from end")

    def display(self):
        if self.head is None:
            print("CLL is empty")
            return
        temp = self.head
        print("Circular Linked List:", end=" ")
        while True:
            print(temp.data, end=" -> ")
            temp = temp.next
            if temp == self.head:
                break
        print("(back to head)")


# ============================================================
# LAB 2 : LINEAR SEARCH
# ============================================================

def linear_search(arr, key):
    for i in range(len(arr)):
        if arr[i] == key:
            return i
    return -1


# ============================================================
# LAB 6 : CIRCULAR QUEUE
# ============================================================

class CircularQueue:
    def __init__(self, size=5):
        self.size = size
        self.arr = [None] * size
        self.front = -1
        self.rear = -1

    def isFull(self):
        return (self.rear + 1) % self.size == self.front

    def isEmpty(self):
        return self.front == -1

    def enqueue(self, value):
        if self.isFull():
            print("Queue Full")
            return
        if self.front == -1:
            self.front = 0
        self.rear = (self.rear + 1) % self.size
        self.arr[self.rear] = value

    def dequeue(self):
        if self.isEmpty():
            print("Queue Empty")
            return None
        val = self.arr[self.front]
        if self.front == self.rear:
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.size
        return val

    def display(self):
        if self.isEmpty():
            print("Queue Empty")
            return
        print("Circular Queue:", end=" ")
        i = self.front
        while True:
            print(self.arr[i], end=" ")
            if i == self.rear:
                break
            i = (i + 1) % self.size
        print()


# ============================================================
# LAB 9 : BUBBLE SORT
# ============================================================

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# ============================================================
# LAB 11 : MERGE SORT (NEWLY ADDED)
# ============================================================

def merge(arr, left, mid, right):
    n1 = mid - left + 1
    n2 = right - mid

    L = arr[left:mid + 1]
    R = arr[mid + 1:right + 1]

    i = j = 0
    k = left

    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1

    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1


def merge_sort(arr, left, right):
    if left < right:
        mid = (left + right) // 2
        merge_sort(arr, left, mid)
        merge_sort(arr, mid + 1, right)
        merge(arr, left, mid, right)


# ============================================================
# LAB 10 : BINARY SEARCH
# ============================================================

def binary_search(arr, key):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            low = mid + 1
        else:
            high = mid - 1
    return -1


# ============================================================
# MAIN DRIVER (Final Combined Execution)
# ============================================================

if __name__ == "__main__":

    print("\n===== CIRCULAR LINKED LIST TEST =====")
    cll = CircularLinkedList()
    for x in [10, 20, 30, 40]:
        cll.insert(x)
    cll.display()

    cll.delete_begin()
    cll.display()

    cll.delete_end()
    cll.display()

    print("\n===== LINEAR SEARCH TEST =====")
    arr = [5, 12, 7, 25, 90]
    print("Search 25 ->", linear_search(arr, 25))

    print("\n===== CIRCULAR QUEUE TEST =====")
    q = CircularQueue()
    q.enqueue(10)
    q.enqueue(20)
    q.enqueue(30)
    q.display()
    q.dequeue()
    q.display()

    print("\n===== BUBBLE SORT TEST =====")
    arr2 = [10, 5, 3, 2, 8]
    print("Sorted:", bubble_sort(arr2))

    print("\n===== MERGE SORT TEST =====")
    arr3 = [38, 27, 43, 3, 9, 82, 10]
    merge_sort(arr3, 0, len(arr3) - 1)
    print("Merge Sorted:", arr3)

    print("\n===== BINARY SEARCH TEST =====")
    sorted_arr = [1, 2, 4, 5, 8, 10]
    print("Search 8 ->", binary_search(sorted_arr, 8))

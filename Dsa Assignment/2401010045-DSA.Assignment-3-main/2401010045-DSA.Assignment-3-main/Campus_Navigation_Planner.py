from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any
import heapq
class Building:
    def __init__(self, building_id: int, name: str, location: str):
        self.id = building_id
        self.name = name
        self.location = location

    def __repr__(self):
        return f"Building(id={self.id}, name='{self.name}', location='{self.location}')"
class BSTNode:
    def __init__(self, building: Building):
        self.building = building
        self.left: Optional[BSTNode] = None
        self.right: Optional[BSTNode] = None

class BinarySearchTree:
    def __init__(self):
        self.root: Optional[BSTNode] = None

    def insert(self, building: Building):
        def _insert(node: Optional[BSTNode], building: Building) -> BSTNode:
            if node is None:
                return BSTNode(building)
            if building.id < node.building.id:
                node.left = _insert(node.left, building)
            elif building.id > node.building.id:
                node.right = _insert(node.right, building)
            else:
                node.building = building
            return node
        self.root = _insert(self.root, building)

    def search(self, building_id: int) -> Optional[Building]:
        node = self.root
        while node:
            if building_id == node.building.id:
                return node.building
            elif building_id < node.building.id:
                node = node.left
            else:
                node = node.right
        return None

    def inorder(self) -> List[Building]:
        res: List[Building] = []
        def _in(node: Optional[BSTNode]):
            if not node: return
            _in(node.left)
            res.append(node.building)
            _in(node.right)
        _in(self.root)
        return res

    def preorder(self) -> List[Building]:
        res: List[Building] = []
        def _pre(node: Optional[BSTNode]):
            if not node: return
            res.append(node.building)
            _pre(node.left)
            _pre(node.right)
        _pre(self.root)
        return res

    def postorder(self) -> List[Building]:
        res: List[Building] = []
        def _post(node: Optional[BSTNode]):
            if not node: return
            _post(node.left)
            _post(node.right)
            res.append(node.building)
        _post(self.root)
        return res

    def height(self) -> int:
        def _h(node: Optional[BSTNode]) -> int:
            if not node: return 0
            return 1 + max(_h(node.left), _h(node.right))
        return _h(self.root)
class AVLNode:
    def __init__(self, building: Building):
        self.building = building
        self.left: Optional[AVLNode] = None
        self.right: Optional[AVLNode] = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root: Optional[AVLNode] = None

    def _get_height(self, node: Optional[AVLNode]) -> int:
        return node.height if node else 0

    def _update_height(self, node: AVLNode):
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

    def _balance_factor(self, node: AVLNode) -> int:
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_right(self, y: AVLNode) -> AVLNode:
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x: AVLNode) -> AVLNode:
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self._update_height(x)
        self._update_height(y)
        return y

    def insert(self, building: Building):
        def _insert(node: Optional[AVLNode], building: Building) -> AVLNode:
            if not node:
                return AVLNode(building)
            if building.id < node.building.id:
                node.left = _insert(node.left, building)
            elif building.id > node.building.id:
                node.right = _insert(node.right, building)
            else:
                node.building = building
                return node

            self._update_height(node)
            balance = self._balance_factor(node)

            
            if balance > 1 and building.id < node.left.building.id:
                return self._rotate_right(node)
            
            if balance < -1 and building.id > node.right.building.id:
                return self._rotate_left(node)
            
            if balance > 1 and building.id > node.left.building.id:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
            
            if balance < -1 and building.id < node.right.building.id:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

            return node
        self.root = _insert(self.root, building)

    def inorder(self) -> List[Building]:
        res = []
        def _in(node: Optional[AVLNode]):
            if not node: return
            _in(node.left)
            res.append(node.building)
            _in(node.right)
        _in(self.root)
        return res

    def height(self) -> int:
        return self._get_height(self.root)
class ExprNode:
    def __init__(self, value: Any, left: 'ExprNode' = None, right: 'ExprNode' = None):
        self.value = value
        self.left = left
        self.right = right

class ExpressionTree:
    def __init__(self):
        self.root: Optional[ExprNode] = None

    @staticmethod
    def from_postfix(tokens: List[str]) -> ExpressionTree:
        stack: List[ExprNode] = []
        for tok in tokens:
            if tok in '+-*/':
                r = stack.pop()
                l = stack.pop()
                stack.append(ExprNode(tok, l, r))
            else:
                
                stack.append(ExprNode(float(tok)))
        tree = ExpressionTree()
        tree.root = stack[-1] if stack else None
        return tree

    def evaluate(self) -> float:
        def _eval(node: ExprNode) -> float:
            if node is None:
                return 0.0
            if node.left is None and node.right is None:
                return node.value
            l = _eval(node.left)
            r = _eval(node.right)
            if node.value == '+':
                return l + r
            if node.value == '-':
                return l - r
            if node.value == '*':
                return l * r
            if node.value == '/':
                return l / r
            raise ValueError('Unknown operator')
        return _eval(self.root)
class Graph:
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj_list: Dict[int, List[Tuple[int, float]]] = {}
        self.nodes_info: Dict[int, Building] = {}

    def add_building(self, building: Building):
        self.nodes_info[building.id] = building
        if building.id not in self.adj_list:
            self.adj_list[building.id] = []

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        if u not in self.adj_list:
            self.adj_list[u] = []
        if v not in self.adj_list:
            self.adj_list[v] = []
        self.adj_list[u].append((v, weight))
        if not self.directed:
            self.adj_list[v].append((u, weight))

    def adjacency_matrix(self) -> Tuple[List[int], List[List[float]]]:
        ids = sorted(self.adj_list.keys())
        index = {nid:i for i,nid in enumerate(ids)}
        n = len(ids)
        mat = [[float('inf')]*n for _ in range(n)]
        for i in range(n):
            mat[i][i] = 0.0
        for u,edges in self.adj_list.items():
            for v,w in edges:
                mat[index[u]][index[v]] = w
        return ids, mat

    def bfs(self, start_id: int) -> List[int]:
        visited = set()
        q = [start_id]
        order = []
        visited.add(start_id)
        while q:
            u = q.pop(0)
            order.append(u)
            for v,_ in self.adj_list.get(u,[]):
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return order

    def dfs(self, start_id: int) -> List[int]:
        visited = set()
        order = []
        def _dfs(u):
            visited.add(u)
            order.append(u)
            for v,_ in self.adj_list.get(u,[]):
                if v not in visited:
                    _dfs(v)
        _dfs(start_id)
        return order

    def dijkstra(self, source_id: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        dist = {nid: float('inf') for nid in self.adj_list}
        prev: Dict[int, Optional[int]] = {nid: None for nid in self.adj_list}
        dist[source_id] = 0.0
        heap = [(0.0, source_id)]
        while heap:
            d,u = heapq.heappop(heap)
            if d>dist[u]:
                continue
            for v,w in self.adj_list.get(u,[]):
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, prev

    def shortest_path(self, source_id: int, target_id: int) -> Tuple[float, List[int]]:
        dist, prev = self.dijkstra(source_id)
        if dist.get(target_id, float('inf'))==float('inf'):
            return float('inf'), []
        path = []
        cur = target_id
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return dist[target_id], path

    def kruskal_mst(self) -> Tuple[float, List[Tuple[int,int,float]]]:
        
        edges = []
        seen = set()
        for u, lst in self.adj_list.items():
            for v,w in lst:
                if (v,u) in seen:
                    continue
                edges.append((w,u,v))
                seen.add((u,v))
        edges.sort()
        parent = {}
        rank = {}
        def make_set(u):
            parent[u]=u
            rank[u]=0
        def find(u):
            if parent[u]!=u:
                parent[u]=find(parent[u])
            return parent[u]
        def union(u,v):
            ru,rv = find(u), find(v)
            if ru==rv: return False
            if rank[ru]<rank[rv]:
                parent[ru]=rv
            else:
                parent[rv]=ru
                if rank[ru]==rank[rv]:
                    rank[ru]+=1
            return True
        for nid in self.adj_list:
            make_set(nid)
        mst_weight = 0.0
        mst_edges: List[Tuple[int,int,float]] = []
        for w,u,v in edges:
            if union(u,v):
                mst_edges.append((u,v,w))
                mst_weight += w
        return mst_weight, mst_edges
def demo():
    print('\n--- Campus Navigation & Utility Planner Demo ---\n')

    
    buildings = [
        Building(10, 'Admin Block', 'Central administration'),
        Building(20, 'Library', 'North wing — books and archives'),
        Building(5, 'Cafeteria', 'Ground floor near east gate'),
        Building(15, 'CompSci Dept', 'Block B, 2nd floor'),
        Building(25, 'Gym', 'South end'),
        Building(12, 'Physics Lab', 'Block C, lab wing')
    ]

    
    bst = BinarySearchTree()
    for b in buildings:
        bst.insert(b)
    print('BST inorder traversal:')
    print(bst.inorder())
    print('BST preorder traversal:')
    print(bst.preorder())
    print('BST postorder traversal:')
    print(bst.postorder())
    print('BST height =', bst.height())

   
    avl = AVLTree()
    for b in buildings:
        avl.insert(b)
    print('\nAVL inorder traversal:')
    print(avl.inorder())
    print('AVL height =', avl.height())
    print('Comparison: AVL height vs BST height ->', avl.height(), 'vs', bst.height())

    
    graph = Graph(directed=False)
    for b in buildings:
        graph.add_building(b)

    
    edges = [
        (10, 20, 120), 
        (10, 5, 60),   
        (5, 15, 200),   
        (20, 12, 150),  
        (12, 15, 80),  
        (15, 25, 300),
        (20, 25, 400)   
    ]
    for u,v,w in edges:
        graph.add_edge(u,v,w)

    print('\nAdjacency List:')
    for k in sorted(graph.adj_list.keys()):
        print(k, '->', graph.adj_list[k])

    ids, mat = graph.adjacency_matrix()
    print('\nAdjacency Matrix (ids order):', ids)
    for row in mat:
        print(row)

   
    start = 10
    print('\nBFS from Admin Block (10):', graph.bfs(start))
    print('DFS from Admin Block (10):', graph.dfs(start))

    
    dist, path = graph.shortest_path(10, 25)
    print('\nShortest path Admin (10) -> Gym (25): distance =', dist, 'path =', path)

    
    mst_weight, mst_edges = graph.kruskal_mst()
    print('\nKruskal MST total weight:', mst_weight)
    print('MST edges:', mst_edges)

    
    postfix = ['100', '0.12', '*', '50', '0.10', '*', '+']
    expr_tree = ExpressionTree.from_postfix(postfix)
    result = expr_tree.evaluate()
    print('\nExpression eval (sample energy bill):', result)

    print('\n--- Demo complete ---\n')

if __name__ == '__main__':
    demo()

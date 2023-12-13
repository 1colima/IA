import heapq

class Node:
    def __init__(self, data, level, fval):
        self.data = data
        self.level = level
        self.fval = fval

    def __lt__(self, other):
        """Método para comparar instâncias de Node."""
        return self.fval < other.fval

    def generate_child(self):
        x, y = self.find(self.data, '_')
        val_list = [[x, y-1], [x, y+1], [x-1, y], [x+1, y]]
        children = []
        for i in val_list:
            child = self.shuffle(self.data, x, y, i[0], i[1])
            if child is not None:
                child_node = Node(child, self.level+1, 0)
                children.append(child_node)
        return children

    def shuffle(self, puz, x1, y1, x2, y2):
        if 0 <= x2 < len(self.data) and 0 <= y2 < len(self.data[0]):
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None

    def copy(self, root):
        temp = [row[:] for row in root]
        return temp

    def find(self, puz, x):
        for i in range(len(puz)):
            for j in range(len(puz[i])):
                if puz[i][j] == x:
                    return i, j

class Puzzle:
    def __init__(self, size):
        self.n = size
        self.priority_queue = []  # Fila de prioridade
        heapq.heapify(self.priority_queue)  # Transforma a lista em uma heap
        self.closed = set()

    def accept(self):
        puz = []
        for i in range(self.n):
            temp = input().split(" ")
            puz.append(temp)
        return puz

    def f(self, start, goal):
        """Função heurística para busca gulosa"""
        return self.h(start.data, goal)

    def h(self, start, goal):
        """Heurística para a busca gulosa"""
        temp = 0
        for i in range(self.n):
            for j in range(len(start[i])):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp

    def process(self):
        """Accept Start and Goal Puzzle state"""
        print("Enter the start state matrix \n")
        start = self.accept()
        print("Enter the goal state matrix \n")        
        goal = self.accept()

        start = Node(start, 0, 0)
        heapq.heappush(self.priority_queue, start)  # Usa a heurística como chave de prioridade
        print("\n\n")
        while self.priority_queue:
            cur = heapq.heappop(self.priority_queue)  # Obtém o nó com menor valor de heurística
            print("")
            print("  | ")
            print("  | ")
            print(" \\\'/ \n")
            for i in cur.data:
                for j in i:
                    print(j, end=" ")
                print("")
            if tuple(map(tuple, cur.data)) == tuple(map(tuple, goal)):
                break
            for i in cur.generate_child():
                child_tuple = tuple(map(tuple, i.data))
                if child_tuple not in self.closed:
                    heapq.heappush(self.priority_queue, i)  # Usa a heurística como chave de prioridade
                    self.closed.add(child_tuple)

puz = Puzzle(3)
puz.process()



import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def criar_rede(vertices, arestas, R, Q):
    d = {
        "V": vertices,
        "A": arestas,
        "R": np.array(R, dtype=float),
        "Q": np.array(Q, dtype=float)
    }
    return d

def construir_C(rede):
    n = len(rede["V"])
    m = len(rede["A"])
    rows, cols, data = [], [], []
    for k, (i, j) in enumerate(rede["A"]):
        rows += [i, j]
        cols += [k, k]
        data += [1, -1]
    return sp.coo_matrix((data, (rows, cols)), shape=(n, m)).tocsr()

def construir_D(rede):
    n = len(rede["V"])
    m = len(rede["A"])
    R = rede["R"]
    rows, cols, data = [], [], []
    for k, (i, j) in enumerate(rede["A"]):
        val = 1.0 / R[k]
        rows += [k, k]
        cols += [i, j]
        data += [-val, val]
    return sp.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()

def montar_Z_com_condicao(C, D, ref_no):
    n, m = C.shape
    # Zera a linha do nó de referência em C
    C_mod = C.tolil()
    C_mod[ref_no, :] = 0
    C_mod = C_mod.tocsr()
    # Matriz F com 1 na posição (ref_no, ref_no)
    F = sp.lil_matrix((n, n))
    F[ref_no, ref_no] = 1.0
    F = F.tocsr()
    I = sp.eye(m, format='csr')
    return sp.bmat([[C_mod, F], [I, D]], format='csr')

def montar_b_com_condicao(Q, m, ref_no):
    b = np.zeros(len(Q) + m)
    b[:len(Q)] = Q
    b[ref_no] = 0.0  # equação substituída
    return b

def resolver_rede(rede, ref_no=0):
    C = construir_C(rede)
    D = construir_D(rede)
    m = len(rede["A"])
    Q = rede["Q"]
    Z = montar_Z_com_condicao(C, D, ref_no)
    b = montar_b_com_condicao(Q, m, ref_no)
    x = spla.spsolve(Z, b)
    return x[:m], x[m:]



def adicionar_ruido(R,len_edges,alfa,chance_mudanca=0.1):
    R_mod = []
    for r in range(len_edges):
        R_mod.append(R[r])

    for k in range(len_edges):
        if np.random.uniform(0,1) < chance_mudanca:
            R_mod[k] *= alfa
    return R_mod

# ------------------------
# Exemplo funcional
# ------------------------
if __name__ == "__main__":
    vertices = [0, 1, 2, 3]
    arestas = [(0, 1), (0, 2), (1, 2), (1, 3)]
    nos_pressao_atm = [3]  # nó 0 fixo
    R = [2, 3, 1, 2]
    Q = [12, 0, 0, 0]  # entra em 0, sai em 3

    # entupimento
    alfa = 10
    chance_entupimento = 0.1

    # p_salvo = {}
    # q_salvo = {}

    p_medio_vetor = {}

    cont = 0
    ensaios = 1000
    p_max_limite = 200

    for j in range(ensaios):
        p_max = 0
        p_min = 0
        R_mod = adicionar_ruido(R, len(arestas), alfa=alfa, chance_mudanca=chance_entupimento)
        rede = criar_rede(vertices, arestas, R_mod, Q)
        q, p = resolver_rede(rede, nos_pressao_atm)
        
        p_max = np.max(p)
        print(j, p_max)

        p_medio_vetor[j] = p.mean()  # usa mean() do numpy
        if p_max> p_max_limite:
            cont+=1
    

    print(f"p_max: {p_max}")
    print(f"p_min: {p_min}")
    print(f"p_medio: {np.mean(list(p_medio_vetor.values()))}")
    print((cont)/ensaios)
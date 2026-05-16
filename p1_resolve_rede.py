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
    for no in ref_no:
        b[no] = 0.0  # equação substituída
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

def calcular_vazoes_saida(rede, q, nos_pressao_atm):
    """
    Calcula a vazão que sai pelos nós de pressão atmosférica.
    
    A vazão de saída em um nó é a soma das vazões que chegam nele
    (considerando a direção do fluxo).
    """
    saidas = {no: 0.0 for no in nos_pressao_atm}
    
    for k, (i, j) in enumerate(rede["A"]):
        qk = q[k]
        
        # Verifica se a vazão está saindo pelos nós atmosféricos
        if i in nos_pressao_atm and qk > 0:  # Vazão saindo do nó i
            saidas[i] += qk
        elif j in nos_pressao_atm and qk < 0:  # Vazão entrando no nó j (negativa significa que vai de j para i)
            saidas[j] += abs(qk)
        elif i in nos_pressao_atm and qk < 0:  # Vazão entrando no nó i (q negativa significa que vai de i para j)
            saidas[i] += abs(qk)
        elif j in nos_pressao_atm and qk > 0:  # Vazão saindo do nó j
            saidas[j] += qk
    
    return saidas

def calcular_vazao_total_saida(saidas):
    """Calcula a vazão total que sai pelos nós atmosféricos"""
    return sum(saidas.values())

def rede_simples():
    vertices = [0, 1, 2, 3]
    arestas = [(0, 1), (0, 2), (1, 2), (1, 3)]
    nos_pressao_atm = [3]  # nó 3 fixo (atmosférico)
    R = [2, 3, 1, 2]
    Q = [12, 0, 0, 0]  # entra em 0, sai em 3
    
    return vertices, arestas, R, Q, nos_pressao_atm

def rede_media():
    vertices = [0, 1, 2, 3, 4, 5, 6, 7]
    arestas = [
        (0, 1), (0, 2), (0, 3),  # conexões do nó fonte
        (1, 4), (1, 5),          # ramificações do nó 1
        (2, 5), (2, 6),          # ramificações do nó 2
        (3, 6), (3, 7),          # ramificações do nó 3
        (4, 5),                  # conexão entre 4 e 5
        (5, 6),                  # conexão entre 5 e 6
        (6, 7)                   # conexão entre 6 e 7
    ]
    nos_pressao_atm = [5, 7]     # nó 5, 7 fixo (atmosférico)
    R = [2.0, 1.5, 2.5,          # resistências das arestas
         1.8, 2.2, 
         1.6, 2.0,
         2.3, 1.9,
         2.1,
         1.7,
         2.4]
    Q = [15, 0, 0, 0, 0, 0, 0, 0]  # vazão de entrada no nó 0
    
    return vertices, arestas, R, Q, nos_pressao_atm

def analisar_rede(nome_rede, vertices, arestas, R, Q, nos_pressao_atm):
    """Função auxiliar para analisar e exibir os resultados da rede"""
    print(f"\n{'='*60}")
    print(f"ANÁLISE DA {nome_rede}")
    print(f"{'='*60}")
    
    rede = criar_rede(vertices, arestas, R, Q)
    q, p = resolver_rede(rede, nos_pressao_atm)
    
    print(f"\nNós de entrada/saída:")
    Q_array = np.array(Q)
    entrada_indices = np.where(Q_array > 0)[0]
    vazao_entrada = Q_array[entrada_indices].sum()
    print(f"Vazão total de entrada: {vazao_entrada:.2f} (nós {entrada_indices.tolist()})")
    print(f"Nós com pressão atmosférica: {nos_pressao_atm}")
    
    print(f"\nVAZÕES NAS ARESTAS:")
    print(f"  {'Aresta':<12} {'Vazão':<10} {'Direção':<10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for k, (i, j) in enumerate(arestas):
        direcao = f"{i}→{j}" if q[k] >= 0 else f"{j}→{i}"
        print(f"  {i}-{j:<9} {abs(q[k]):<10.2f} {direcao:<10}")
    
    print(f"\nPRESSÕES NOS NÓS:")
    print(f"  {'Nó':<6} {'Pressão':<10} {'Status':<15}")
    print(f"  {'-'*6} {'-'*10} {'-'*15}")
    for i, pressao in enumerate(p):
        status = "(exposto à atm)" if i in nos_pressao_atm else "-"
        print(f"  {i:<6} {pressao:<10.2f} {status}")
    
    # Calcular vazões de saída
    saidas = calcular_vazoes_saida(rede, q, nos_pressao_atm)
    vazao_total_saida = calcular_vazao_total_saida(saidas)
    vazao_total_entrada = vazao_entrada
    
    print(f"\nVAZÕES DE SAÍDA (pelos nós atmosféricos):")
    for no, vazao in saidas.items():
        print(f"  Nó {no}: {vazao:.2f}")
    
    print(f"\nBALANÇO HÍDRICO:")
    print(f"  Vazão total de entrada: {vazao_total_entrada:.2f}")
    print(f"  Vazão total de saída:   {vazao_total_saida:.2f}")
    print(f"  Diferença:              {vazao_total_entrada - vazao_total_saida:.2f}")
    
    if abs(vazao_total_entrada - vazao_total_saida) < 1e-6:
        print(f"Balanço hídrico ok!")
    else:
        print(f"Balanço hídrico com discrepância!")

# ------------------------
# Exemplo funcional
# ------------------------
if __name__ == "__main__":
    # Teste com rede simples
    vertices, arestas, R, Q, nos_pressao_atm = rede_simples()
    analisar_rede("REDE SIMPLES", vertices, arestas, R, Q, nos_pressao_atm)
    
    # Teste com rede média
    vertices, arestas, R, Q, nos_pressao_atm = rede_media()
    analisar_rede("REDE MÉDIA", vertices, arestas, R, Q, nos_pressao_atm)
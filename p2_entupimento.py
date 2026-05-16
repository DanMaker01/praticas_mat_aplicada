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

def montar_Z_com_condicao(C, D, ref_nos):
    """
    Monta a matriz Z do sistema linear.
    
    ref_nos: lista de nós com pressão fixa (nós expostos à atmosfera)
    """
    n, m = C.shape
    
    # Converte para lista se for inteiro (para compatibilidade)
    if isinstance(ref_nos, int):
        ref_nos = [ref_nos]
    
    # Modifica C: zera as linhas dos nós de referência
    C_mod = C.tolil()
    for no in ref_nos:
        C_mod[no, :] = 0
    C_mod = C_mod.tocsr()
    
    # Matriz F: identidade para os nós de referência
    F = sp.lil_matrix((n, n))
    for no in ref_nos:
        F[no, no] = 1.0
    F = F.tocsr()
    
    I = sp.eye(m, format='csr')
    return sp.bmat([[C_mod, F], [I, D]], format='csr')

def montar_b_com_condicao(Q, m, ref_nos):
    """
    Monta o vetor b do sistema linear.
    
    ref_nos: lista de nós com pressão fixa
    """
    # Converte para lista se for inteiro (para compatibilidade)
    if isinstance(ref_nos, int):
        ref_nos = [ref_nos]
    
    b = np.zeros(len(Q) + m)
    b[:len(Q)] = Q
    
    # Para os nós de referência, a equação de continuidade é substituída
    # pela condição de pressão fixa (p = 0)
    for no in ref_nos:
        b[no] = 0.0  # p = 0
    
    return b

def resolver_rede(rede, ref_nos):
    """
    Resolve a rede hidráulica.
    
    ref_nos: lista de nós com pressão fixa (nós expostos à atmosfera)
    """
    C = construir_C(rede)
    D = construir_D(rede)
    m = len(rede["A"])
    Q = rede["Q"]
    
    Z = montar_Z_com_condicao(C, D, ref_nos)
    b = montar_b_com_condicao(Q, m, ref_nos)
    
    try:
        x = spla.spsolve(Z, b)
        return x[:m], x[m:]
    except Exception as e:
        print(f"Erro na solução: {e}")
        return np.zeros(m), np.zeros(len(rede["V"]))

def adicionar_ruido(R, len_edges, alfa, chance_mudanca=0.1):
    """
    Adiciona ruído (entupimento) às resistências da rede.
    
    Parâmetros:
    - R: lista de resistências originais
    - len_edges: número de arestas
    - alfa: fator de multiplicação para entupimento (ex: 100 = aumenta 100x)
    - chance_mudanca: probabilidade de cada aresta sofrer entupimento
    
    Retorna:
    - R_mod: lista de resistências modificadas
    """
    R_mod = R.copy()  # Usa copy() em vez de criar lista manualmente
    
    for k in range(len_edges):
        if np.random.uniform(0, 1) < chance_mudanca:
            R_mod[k] *= alfa
    
    return R_mod

def rede_simples():
    vertices = [0, 1, 2, 3]
    arestas = [(0, 1), (0, 2), (1, 2), (1, 3)]
    nos_pressao_atm = [3]  # nó 3 fixo (atmosférico)
    R = [2, 3, 1, 2]
    Q = [12, 0, 0, 0]  # entra em 0
    
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
    nos_pressao_atm = [5, 7]     # nós 5 e 7 fixos (atmosféricos)
    R = [2.0, 1.5, 2.5,          # resistências das arestas
         1.8, 2.2, 
         1.6, 2.0,
         2.3, 1.9,
         2.1,
         1.7,
         2.4]
    Q = [15, 0, 0, 0, 0, 0, 0, 0]  # vazão de entrada no nó 0
    
    return vertices, arestas, R, Q, nos_pressao_atm

def analisar_entupimentos(rede_func, nome_rede="Rede", qtd_ensaios=100, p_max_limite=200, 
                          alfa=10e2, chance_entupimento=0.1):
    """
    Analisa a probabilidade de estouro de pressão devido a entupimentos.
    
    Parâmetros:
    - rede_func: função que retorna (vertices, arestas, R, Q, nos_pressao_atm)
    - nome_rede: nome da rede para exibição
    - qtd_ensaios: número de ensaios (cenários de entupimento)
    - p_max_limite: pressão máxima permitida
    - alfa: fator de entupimento
    - chance_entupimento: probabilidade de cada aresta entupir (r)
    """
    print(f"\n{'='*70}")
    print(f"ANÁLISE DE ENTUPIMENTOS - {nome_rede}")
    print(f"{'='*70}")
    print(f"Configurações:")
    print(f"  - Fator de entupimento (α): {alfa}")
    print(f"  - Chance de entupimento (r): {chance_entupimento}")
    print(f"  - Pressão máxima limite: {p_max_limite}")
    print(f"  - Número de ensaios: {qtd_ensaios}")
    
    # Carrega a rede base
    vertices, arestas, R_base, Q, nos_pressao_atm = rede_func()
    
    # Calcula número de simulações por ensaio (proporcional ao tamanho da rede)
    qtd_redes_com_perturbacao = 100 + 40 * int(np.sqrt(len(arestas) * len(vertices)))
    #justificar
    print(f"  - Simulações de rede por ensaio: {qtd_redes_com_perturbacao}")
    print(f"{'='*70}")
    
    historico = {}
    
    for i in range(qtd_ensaios):
        cont_redes_pressao_estourou = 0
        pressoes_maximas = []
        pressoes_medias = []
        
        for j in range(qtd_redes_com_perturbacao):
            # Aplica entupimento aleatório
            R_mod = adicionar_ruido(R_base, len(arestas), alfa=alfa, 
                                   chance_mudanca=chance_entupimento)
            
            # Resolve a rede com resistências modificadas
            rede = criar_rede(vertices, arestas, R_mod, Q)
            q, p = resolver_rede(rede, nos_pressao_atm)
            
            # Seleciona valores importantes
            maior_pressao_na_rede = np.max(p)
            media_pressao_na_rede = np.mean(p)
            
            pressoes_maximas.append(maior_pressao_na_rede)
            pressoes_medias.append(media_pressao_na_rede)
            
            if maior_pressao_na_rede > p_max_limite:
                cont_redes_pressao_estourou += 1
        
        # Calcula estatísticas do ensaio
        prob_estouro = float(cont_redes_pressao_estourou) / float(qtd_redes_com_perturbacao)
        historico[i] = prob_estouro
        
        print(f"Ensaio {i+1:3d}/{qtd_ensaios} | Estouros: {cont_redes_pressao_estourou:4d}/{qtd_redes_com_perturbacao:4d} "
              f"({100*prob_estouro:5.2f}%)")
    
    # Análise dos ensaios
    valores = list(historico.values())
    media = np.mean(valores)
    desvio = np.std(valores)
    maximo = np.max(valores)
    minimo = np.min(valores)
    
    print(f"{'='*70}")
    print(f"RESULTADOS - {nome_rede}")
    print(f"{'='*70}")
    print(f"Probabilidade de estouro de pressão (p > {p_max_limite}):")
    print(f"  - Média:      {100*media:.2f}%")
    print(f"  - Desvio padrão: ±{100*desvio:.2f}%")
    print(f"  - Mínimo:     {100*minimo:.2f}%")
    print(f"  - Máximo:     {100*maximo:.2f}%")
    print(f"{'='*70}")
    
    return historico, media, desvio

if __name__ == "__main__":
    print("P2 ENTUPIMENTO")
    # Configurações
    ALFA = 10e2  # Fator de entupimento (aumenta resistência em 1000x)
    CHANCE_ENTUPIMENTO = 0.1  # Probabilidade de cada aresta entupir (r)
    QTD_ENSAIOS = 100
    P_MAX_LIMITE = 200
    
    # Teste com rede simples
    historico_simples, media_simples, desvio_simples = analisar_entupimentos(
        rede_func=rede_simples,
        nome_rede="REDE SIMPLES (4 nós, 4 arestas)",
        qtd_ensaios=QTD_ENSAIOS,
        p_max_limite=P_MAX_LIMITE,
        alfa=ALFA,
        chance_entupimento=CHANCE_ENTUPIMENTO
    )
    
    # Teste com rede média
    historico_media, media_media, desvio_media = analisar_entupimentos(
        rede_func=rede_media,
        nome_rede="REDE MÉDIA (8 nós, 12 arestas)",
        qtd_ensaios=QTD_ENSAIOS,
        p_max_limite=P_MAX_LIMITE,
        alfa=ALFA,
        chance_entupimento=CHANCE_ENTUPIMENTO
    )
    
    # Comparação final
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO ENTRE REDES")
    print(f"{'='*70}")
    print(f"Rede Simples:  {100*media_simples:.2f}% ± {100*desvio_simples:.2f}%")
    print(f"Rede Média:    {100*media_media:.2f}% ± {100*desvio_media:.2f}%")
    
    if media_simples > media_media:
        print(f"\nA rede simples tem MAIOR probabilidade de estouro!")
    else:
        print(f"\nA rede média é mais robusta contra entupimentos!")
    print(f"{'='*70}")
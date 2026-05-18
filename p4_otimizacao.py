import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import random
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

# ============================
# CONFIGURAÇÃO DE PLOT
# ============================

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# ============================
# SOLVER HIDRÁULICO
# ============================

def resolver_rede(vertices, arestas, R, Q, ref_nos):
    n, m = len(vertices), len(arestas)
    
    rows, cols, data = [], [], []
    for k, (i, j) in enumerate(arestas):
        rows += [i, j]
        cols += [k, k]
        data += [1, -1]
    C = sp.coo_matrix((data, (rows, cols)), shape=(n, m)).tocsr()
    
    rows, cols, data = [], [], []
    for k, (i, j) in enumerate(arestas):
        val = 1.0 / R[k]
        rows += [k, k]
        cols += [i, j]
        data += [-val, val]
    D = sp.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()
    
    if isinstance(ref_nos, int):
        ref_nos = [ref_nos]
    
    C_mod = C.tolil()
    F = sp.lil_matrix((n, n))
    for no in ref_nos:
        C_mod[no, :] = 0
        F[no, no] = 1.0
    
    Z = sp.bmat([[C_mod.tocsr(), F.tocsr()], [sp.eye(m), D]], format='csr')
    b = np.zeros(n + m)
    b[:n] = Q
    for no in ref_nos:
        b[no] = 0.0
    
    try:
        x = spla.spsolve(Z, b)
        return x[:m], x[m:]
    except:
        return np.zeros(m), np.zeros(n)

# ============================
# REDES ESTRUTURADAS
# ============================

def rede_media():
    vertices = [0, 1, 2, 3, 4, 5, 6, 7]
    arestas = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 5), (2, 6), (3, 6), (3, 7), (4, 5), (5, 6), (6, 7)]
    nos_pressao_atm = [5, 7]
    R = [2.0, 1.5, 2.5, 1.8, 2.2, 1.6, 2.0, 2.3, 1.9, 2.1, 1.7, 2.4]
    Q = [15, 0, 0, 0, 0, 0, 0, 0]
    return vertices, arestas, R, Q, nos_pressao_atm

def rede_grande():
    vertices = list(range(12))
    arestas = []
    
    for i in range(4):
        for j in range(2):
            no1 = i * 3 + j
            no2 = i * 3 + j + 1
            arestas.append((no1, no2))
    
    for j in range(3):
        for i in range(3):
            no1 = i * 3 + j
            no2 = (i + 1) * 3 + j
            arestas.append((no1, no2))
    
    for i in range(3):
        for j in range(2):
            no1 = i * 3 + j
            no2 = (i + 1) * 3 + j + 1
            arestas.append((no1, no2))
            no1 = i * 3 + j + 1
            no2 = (i + 1) * 3 + j
            arestas.append((no1, no2))
    
    nos_pressao_atm = [3, 7, 11]
    m = len(arestas)
    R = [2.0] * m
    Q = [0] * len(vertices)
    Q[0] = 30
    
    return vertices, arestas, R, Q, nos_pressao_atm

def rede_anel():
    vertices = list(range(8))
    arestas = [(i, (i+1) % 8) for i in range(8)]
    arestas += [(0, 4), (1, 5), (2, 6), (3, 7)]
    arestas += [(0, 2), (2, 4), (4, 6), (6, 0)]
    nos_pressao_atm = [4, 7]
    m = len(arestas)
    R = [1.5] * m
    Q = [0] * len(vertices)
    Q[0] = 20
    return vertices, arestas, R, Q, nos_pressao_atm

# ============================
# VISUALIZAÇÃO DA REDE
# ============================

def visualizar_rede(vertices, arestas, R, Q, p, q, ref_nos, titulo="Rede Hidráulica", 
                     arestas_destacadas=None, salvar=False):
    """Visualiza a rede hidráulica com todas as informações"""
    
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(arestas)
    
    if len(vertices) <= 8:
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    
    # Cores dos nós
    node_colors = []
    node_sizes = []
    for i in vertices:
        if i in ref_nos:
            node_colors.append('#4CAF50')
            node_sizes.append(800)
        elif Q[i] > 0:
            node_colors.append('#FF9800')
            node_sizes.append(900)
        else:
            pressao = p[i] if i < len(p) else 0
            intensity = min(0.8, max(0.2, pressao / 20))
            node_colors.append(plt.cm.RdYlBu_r(intensity))
            node_sizes.append(700)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=node_sizes, ax=ax, alpha=0.9)
    
    # Arestas
    edge_colors = []
    edge_widths = []
    edge_styles = []
    
    for k, (i, j) in enumerate(arestas):
        if k < len(q):
            edge_colors.append('#2196F3' if q[k] > 0 else '#F44336')
        else:
            edge_colors.append('#9E9E9E')
        
        if k < len(R):
            width = max(1, min(6, 15 / R[k]))
            edge_widths.append(width)
        else:
            edge_widths.append(2)
        
        if arestas_destacadas and k in arestas_destacadas:
            edge_styles.append('dashed')
        else:
            edge_styles.append('solid')
    
    for k, (i, j) in enumerate(arestas):
        style = edge_styles[k]
        if style == 'dashed':
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], 
                                   edge_color=[edge_colors[k]], 
                                   width=edge_widths[k], 
                                   style='dashed', ax=ax, alpha=0.8)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], 
                                   edge_color=[edge_colors[k]], 
                                   width=edge_widths[k], 
                                   ax=ax, alpha=0.7)
    
    # Labels dos nós
    node_labels = {}
    for i in vertices:
        if i < len(p) and i < len(Q):
            pressao = p[i]
            vazao = Q[i]
            if i in ref_nos:
                label = f'{i}\nATM\np={pressao:.1f}'
            elif vazao > 0:
                label = f'{i}\nFONTE\nQ={vazao:.1f}\np={pressao:.1f}'
            else:
                label = f'{i}\np={pressao:.1f}'
        else:
            label = f'{i}'
        node_labels[i] = label
    
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=ax)
    
    # Labels das arestas
    edge_labels = {}
    for k, (i, j) in enumerate(arestas):
        if k < len(R) and k < len(q):
            r_val = R[k]
            q_val = q[k]
            seta = '→' if q_val > 0 else '←'
            edge_labels[(i, j)] = f'R={r_val:.3f}\nq={abs(q_val):.1f}{seta}'
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Nó Atmosférico (p=0)'),
        Patch(facecolor='#FF9800', label='Nó Fonte (Q>0)'),
        Patch(facecolor='#2196F3', label='Fluxo Positivo'),
        Patch(facecolor='#F44336', label='Fluxo Negativo'),
        Patch(facecolor='#9E9E9E', label='Sem Fluxo')
    ]
    if arestas_destacadas:
        legend_elements.append(Patch(facecolor='none', edgecolor='black', 
                                     linestyle='dashed', label='Aresta Reforçada'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Informações adicionais
    if len(p) > 0:
        mascara = [i for i in range(len(p)) if i not in ref_nos]
        if len(mascara) > 0:
            p_validas = p[mascara]
            info_text = f'Pressão: max={np.max(p_validas):.2f}, min={np.min(p_validas):.2f}, média={np.mean(p_validas):.2f}'
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_axis_off()
    plt.tight_layout()
    
    if salvar:
        plt.savefig(f"{titulo.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    
    plt.show()

def comparar_redes(vertices, arestas, R_base, Q, ref_nos, x_reforcos, 
                   resistencia_nova=0.01, titulo="Comparação"):
    """
    Mostra antes e depois da aplicação dos reforços
    """
    m = len(arestas)
    
    # Rede original
    q_orig, p_orig = resolver_rede(vertices, arestas, R_base, Q, ref_nos)
    
    # Rede com reforços (SUBSTITUI a resistência por resistencia_nova)
    R_mod = R_base.copy()
    arestas_reforcadas = []
    for k in range(m):
        if x_reforcos[k] == 1:
            R_mod[k] = resistencia_nova
            arestas_reforcadas.append(k)
    
    q_mod, p_mod = resolver_rede(vertices, arestas, R_mod, Q, ref_nos)
    
    # Cálculo das diferenças
    mascara = [i for i in range(len(vertices)) if i not in ref_nos]
    dif_orig = np.max(p_orig[mascara]) - np.min(p_orig[mascara])
    dif_mod = np.max(p_mod[mascara]) - np.min(p_mod[mascara])
    melhora = (dif_orig - dif_mod) / dif_orig * 100 if dif_orig > 0 else 0
    
    # Cria figura com dois subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"{titulo} - Melhora: {melhora:.1f}%", fontsize=14, fontweight='bold')
    
    for ax, tit, R_use, q_use, p_use, is_original in zip(
        axes, 
        ["ANTES (Rede Original)", f"DEPOIS ({len(arestas_reforcadas)} canos grossos, R={resistencia_nova})"],
        [R_base, R_mod],
        [q_orig, q_mod],
        [p_orig, p_mod],
        [True, False]
    ):
        G = nx.Graph()
        G.add_nodes_from(vertices)
        G.add_edges_from(arestas)
        
        if len(vertices) <= 8:
            pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        else:
            pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)
        
        # Cores dos nós
        node_colors = []
        for i in vertices:
            if i in ref_nos:
                node_colors.append('#4CAF50')
            elif Q[i] > 0:
                node_colors.append('#FF9800')
            else:
                pressao = p_use[i] if i < len(p_use) else 0
                intensity = min(0.8, max(0.2, pressao / 20))
                node_colors.append(plt.cm.RdYlBu_r(intensity))
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                               node_size=700, ax=ax, alpha=0.9)
        
        # Arestas
        edge_colors = []
        edge_widths = []
        edge_styles = []
        
        for k, (i, j) in enumerate(arestas):
            if k < len(q_use):
                edge_colors.append('#2196F3' if q_use[k] > 0 else '#F44336')
            else:
                edge_colors.append('#9E9E9E')
            
            if k < len(R_use):
                # Para arestas reforçadas, usa largura maior (menor resistência)
                if not is_original and k in arestas_reforcadas:
                    width = 6  # Aresta grossa para cano reforçado
                else:
                    width = max(1, min(5, 15 / R_use[k]))
                edge_widths.append(width)
            else:
                edge_widths.append(2)
            
            if not is_original and k in arestas_reforcadas:
                edge_styles.append('dashed')
            else:
                edge_styles.append('solid')
        
        for k, (i, j) in enumerate(arestas):
            style = edge_styles[k]
            if style == 'dashed':
                nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], 
                                       edge_color=[edge_colors[k]], 
                                       width=edge_widths[k], 
                                       style='dashed', ax=ax, alpha=0.8)
            else:
                nx.draw_networkx_edges(G, pos, edgelist=[(i, j)], 
                                       edge_color=[edge_colors[k]], 
                                       width=edge_widths[k], 
                                       ax=ax, alpha=0.7)
        
        # Labels dos nós
        node_labels = {}
        for i in vertices:
            if i < len(p_use) and i < len(Q):
                pressao = p_use[i]
                vazao = Q[i]
                if i in ref_nos:
                    label = f'{i}\nATM\np={pressao:.1f}'
                elif vazao > 0:
                    label = f'{i}\nQ={vazao:.1f}\np={pressao:.1f}'
                else:
                    label = f'{i}\np={pressao:.1f}'
            else:
                label = f'{i}'
            node_labels[i] = label
        
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=ax)
        
        # Labels das arestas
        edge_labels = {}
        for k, (i, j) in enumerate(arestas):
            if k < len(R_use) and k < len(q_use):
                r_val = R_use[k]
                q_val = q_use[k]
                seta = '→' if q_val > 0 else '←'
                # Formato com 3 casas decimais para R pequeno
                edge_labels[(i, j)] = f'R={r_val:.3f}\nq={abs(q_val):.1f}{seta}'
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
        
        ax.set_title(tit, fontsize=12, fontweight='bold')
        ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    
    # Print dos resultados
    print(f"Resultados Comparação:")
    print(f"Arestas reforçadas: {arestas_reforcadas}")
    # print(f"Resistência dos canos grossos: {resistencia_nova}")
    print(f"\nPressões (nós não-atmosféricos):")
    print(f"- Original:   max={np.max(p_orig[mascara]):.2f}, min={np.min(p_orig[mascara]):.2f}, dif={dif_orig:.4f}")
    print(f"- Modificada: max={np.max(p_mod[mascara]):.2f}, min={np.min(p_mod[mascara]):.2f}, dif={dif_mod:.4f}")
    print(f"- Melhora: {melhora:.1f}%")
    
    return p_orig, p_mod, q_orig, q_mod

# ============================
# BRKGA ROBUSTO
# ============================

def brkga(m, T, fitness_func, pop_size=80, elite_size=12, 
                   mut_prob=0.5, n_geracoes=100, renovacao_interval=25, 
                   taxa_renovacao=0.3, verbose=False):
    
    def decode(cromo):
        x = np.zeros(m)
        x[np.argsort(cromo)[-T:]] = 1
        return x
    
    pop = np.random.rand(pop_size, m)
    melhor_solucao = None
    melhor_fitness = -np.inf
    estagnacao = 0
    
    for gen in range(n_geracoes):
        solucoes = []
        fitness = []
        for cromo in pop:
            x = decode(cromo)
            solucoes.append(x)
            fit = fitness_func(x)
            fitness.append(fit)
        
        fitness = np.array(fitness)
        idx = np.argsort(fitness)[::-1]
        pop = pop[idx]
        fitness = fitness[idx]
        solucoes = [solucoes[i] for i in idx]
        
        if fitness[0] > melhor_fitness:
            melhor_fitness = fitness[0]
            melhor_solucao = solucoes[0].copy()
            estagnacao = 0
        else:
            estagnacao += 1
        
        if verbose and (gen % 20 == 0):
            print(f"    Gen {gen:3d}: fit={fitness[0]:.4f}")
        
        if (gen > 0 and gen % renovacao_interval == 0) or estagnacao > 30:
            n_manter = int(pop_size * (1 - taxa_renovacao))
            pop[:n_manter] = pop[:n_manter]
            pop[n_manter:] = np.random.rand(pop_size - n_manter, m)
            estagnacao = 0
        
        nova_pop = pop[:elite_size].copy()
        
        while len(nova_pop) < pop_size:
            if random.random() < 0.7:
                pai1 = pop[random.randint(0, elite_size - 1)]
            else:
                pai1 = pop[random.randint(elite_size, pop_size - 1)]
            
            pai2 = pop[random.randint(elite_size, pop_size - 1)]
            filho = pai1 * 0.6 + pai2 * 0.4
            
            if random.random() < mut_prob:
                intensidade = 0.25 * (1 - gen / n_geracoes) + 0.1
                filho += np.random.normal(0, intensidade, m)
            
            nova_pop = np.vstack([nova_pop, np.clip(filho, 0, 1)])
        
        pop = nova_pop
    
    return melhor_solucao, melhor_fitness

def brkga_multiplas_execucoes(m, T, fitness_func, n_execucoes=3, **kwargs):
    melhores_resultados = []
    
    for execucao in range(n_execucoes):
        seed = execucao * 12345
        random.seed(seed)
        np.random.seed(seed)
        
        melhor_x, melhor_fit = brkga(m, T, fitness_func, 
                                              verbose=False, **kwargs)
        
        melhores_resultados.append({
            'execucao': execucao,
            'x': melhor_x,
            'fitness': melhor_fit
        })
    
    melhor_execucao = max(melhores_resultados, key=lambda r: r['fitness'])
    return melhor_execucao['x'], melhor_execucao['fitness']

def problema_reforcos(vertices, arestas, R_base, Q, ref_nos, resistencia_de_T=0.01):
    m = len(arestas)
    
    _, p_orig = resolver_rede(vertices, arestas, R_base, Q, ref_nos)
    mascara = [i for i in range(len(p_orig)) if i not in ref_nos]
    dif_orig = np.max(p_orig[mascara]) - np.min(p_orig[mascara])
    
    def fitness(x):
        R_mod = R_base.copy()
        for k in range(m):
            if x[k] == 1:
                R_mod[k] = resistencia_de_T
        
        _, p = resolver_rede(vertices, arestas, R_mod, Q, ref_nos)
        p_val = p[mascara]
        
        if len(p_val) == 0 or np.any(np.isnan(p_val)):
            return -np.inf
        
        diferenca = np.max(p_val) - np.min(p_val)
        melhora = (dif_orig - diferenca) / dif_orig if dif_orig > 0 else 0
        
        penalidade = 0
        if np.min(p_val) < 2.0:
            penalidade += 3.0 * (2.0 - np.min(p_val))
        if np.max(p_val) > 35.0:
            penalidade += 1.0 * (np.max(p_val) - 35.0)
        
        return melhora - 0.05 * penalidade
    
    return fitness



if __name__ == "__main__":
    
    # Parâmetros
    RESISTENCIA_CANO_GROSSO = 0.01  # Resistência muito baixa para canos grossos
    
    BRKGA_PARAMS = {
        'pop_size': 60,
        'elite_size': 10,
        'mut_prob': 0.5,
        'n_geracoes': 60,
        'renovacao_interval': 20,
        'taxa_renovacao': 0.3
    }
    
    
    vertices, arestas, R_base, Q, ref_nos = rede_grande()
    nome = "Rede Grande"
    T = 2

    print(f"Rede selecionada: {nome}")
    print(f"Nós: {len(vertices)}, Arestas: {len(arestas)}")
    print(f"Nós atmosféricos: {ref_nos}")
    print(f"Quantidade de canos grossos: {T}")
    print(f"Resistência do cano grosso: {RESISTENCIA_CANO_GROSSO}")
    
    q_orig, p_orig = resolver_rede(vertices, arestas, R_base, Q, ref_nos)
    
    fitness_func = problema_reforcos(vertices, arestas, R_base, Q, ref_nos, 
                                      resistencia_de_T=RESISTENCIA_CANO_GROSSO)
    
    melhor_x, melhor_fit = brkga_multiplas_execucoes(
        len(arestas), T, fitness_func, 
        n_execucoes=3,
        **BRKGA_PARAMS
    )
    comparar_redes(vertices, arestas, R_base, Q, ref_nos, melhor_x, 
                   resistencia_nova=RESISTENCIA_CANO_GROSSO,
                   titulo=f"{nome} - T={T}")
    
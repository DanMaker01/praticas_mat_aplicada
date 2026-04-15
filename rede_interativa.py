# implementar a vazão de saída, que não é calculada diretamente pelo sistema

import pygame
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import random
from collections import deque

pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rede Hidráulica Interativa")
font = pygame.font.SysFont("Arial", 14)
clock = pygame.time.Clock()

DEBUG = False

# ============================
# VARIÁVEIS GLOBAIS
# ============================
selected_node = None
selected_edge = None
creating_edge = None

# ============================
# MODELO MATEMÁTICO
# ============================
def criar_rede(vertices, arestas, R, Q) -> dict:
    return {"V": vertices, "A": arestas, "R": np.array(R, dtype=float), "Q": np.array(Q, dtype=float)}

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

def montar_Z(C, D, ref_no=[0]):
    n, m = C.shape
    C_mod = C.tolil()
    for r in ref_no:
        C_mod[r, :] = 0
    C_mod = C_mod.tocsr()

    F = sp.lil_matrix((n, n))
    for r in ref_no:
        F[r, r] = 1.0
    F = F.tocsr()

    I = sp.eye(m, format='csr')
    return sp.bmat([[C_mod, F], [I, D]], format='csr')

def montar_b(Q, m, ref_no=[0]):
    b = np.zeros(len(Q) + m)
    b[:len(Q)] = Q
    for r in ref_no:
        b[r] = 0.0
    return b

def resolver_rede(rede, ref_no=[0]):
    if len(rede["A"]) == 0:
        return np.array([]), np.zeros(len(rede["V"]))

    C = construir_C(rede)
    D = construir_D(rede)
    Z = montar_Z(C, D, ref_no)
    b = montar_b(rede["Q"], len(rede["A"]), ref_no)

    try:
        x = spla.spsolve(Z, b)
        return x[:len(rede["A"])], x[len(rede["A"]):]
    except Exception:
        # Se houver erro, retorna vazões nulas e pressões zero
        return np.zeros(len(rede["A"])), np.zeros(len(rede["V"]))

# ============================
# GERADOR ALEATÓRIO
# ============================
num_nos_min, num_nos_max = 5, 10
num_arestas_min, num_arestas_max = 6, 12
R_min, R_max = 1.0, 5.0
Qin_min, Qin_max = 5, 20
x_min, x_max = 100, 900
y_min, y_max = 100, 500
NUM_PARTICLES = 5

def gerar_rede_1():
    global nodes, edges, R, Q, pressao_atm, flow_particles

    num_nos = 0
    nodes = []
    nos = list(range(num_nos))
    arestas = set()
    edges = list(arestas)
    R = [1 for _ in edges]
    Q = [0]*num_nos
    Qin = 10
    pressao_atm = []


    flow_particles = [[p/NUM_PARTICLES for p in range(NUM_PARTICLES)] for _ in edges]

def adicionar_ruido(R,alfa,chance_mudanca=0.1):
    for k in range(len(edges)):
        if random.uniform(0,1) < chance_mudanca:
            R[k] *= alfa


def gerar_rede_aleatoria():
    global nodes, edges, R, Q, pressao_atm, flow_particles

    num_nos = random.randint(num_nos_min, num_nos_max)
    nodes = [{"pos": (random.randint(x_min, x_max), random.randint(y_min, y_max))} for _ in range(num_nos)]

    nos = list(range(num_nos))
    arestas = set()
    for i in range(1, num_nos):
        j = random.randint(0, i - 1)
        arestas.add((j, i))

    max_arestas = num_nos * (num_nos - 1) // 2
    num_arestas_desejado = min(random.randint(num_arestas_min, num_arestas_max), max_arestas)
    while len(arestas) < num_arestas_desejado:
        i, j = random.sample(nos, 2)
        arestas.add(tuple(sorted((i, j))))

    edges = list(arestas)
    R = [np.random.uniform(R_min, R_max) for _ in edges]
    Q = [0]*num_nos
    Qin = np.random.uniform(Qin_min, Qin_max)           
    no_fonte = random.choice(nos)
    Q[no_fonte] = Qin
    no_saida = random.choice([n for n in nos if n != no_fonte])
    pressao_atm = [no_saida]

    flow_particles = [[p/NUM_PARTICLES for p in range(NUM_PARTICLES)] for _ in edges]

# ============================
# DETEÇÃO DE NÓ E ARESTA
# ============================
def get_node_at(pos):
    for i, n in enumerate(nodes):
        if np.linalg.norm(np.array(n["pos"]) - pos) < 12:
            return i
    return None

def get_edge_at(pos):
    for k, (i, j) in enumerate(edges):
        p1 = np.array(nodes[i]['pos'], float)
        p2 = np.array(nodes[j]['pos'], float)
        line_vec = p2 - p1
        point_vec = pos - p1
        line_len = np.linalg.norm(line_vec)
        if line_len == 0: continue
        line_dir = line_vec / line_len
        proj = np.dot(point_vec, line_dir)
        if 0 <= proj <= line_len:
            closest = p1 + proj*line_dir
            dist = np.linalg.norm(pos - closest)
            if dist < 6:
                return k
    return None

# ============================
# DESENHO
# ============================

def sanitize_nodes():
    """Corrige posições de nós que são NaN ou infinitas."""
    for i, node in enumerate(nodes):
        pos = node["pos"]
        if not np.isfinite(pos).all():
            node["pos"] = (0, 0)

def reset_particles():
    """Recria todas as partículas com valores iniciais."""
    global flow_particles
    flow_particles = [[p / NUM_PARTICLES for p in range(NUM_PARTICLES)] for _ in edges]

def draw_flow_particles(q):
    # Garante que todos os nós tenham posições válidas
    sanitize_nodes()

    for k, (i, j) in enumerate(edges):
        if k >= len(q):
            continue

        # Se a vazão for NaN, não desenha partículas nesta aresta
        if np.isnan(q[k]):
            continue

        p1 = np.array(nodes[i]['pos'], float)
        p2 = np.array(nodes[j]['pos'], float)

        # Verifica posições dos nós
        if not np.isfinite(p1).all() or not np.isfinite(p2).all():
            continue

        d = p2 - p1
        L = np.linalg.norm(d)
        if L == 0:
            continue

        d /= L
        speed = q[k] * 0.001

        for idx in range(len(flow_particles[k])):
            # Atualiza posição da partícula
            flow_particles[k][idx] = (flow_particles[k][idx] + speed) % 1.0
            pos = p1 + flow_particles[k][idx] * d * L

            # Desenha se a posição for válida e estiver dentro da tela
            if np.isfinite(pos).all() and 0 <= pos[0] <= WIDTH and 0 <= pos[1] <= HEIGHT:
                color = (0, 255, 255) if q[k] >= 0 else (255, 100, 100)
                pygame.draw.circle(screen, color, pos.astype(int), 4)

def draw_debug_info(q, p):
    """Desenha informações de debug no canto superior esquerdo"""
    y_offset = 30
    
    # Título do debug
    title = font.render("=== DEBUG MODE ===", True, (255, 255, 0))
    screen.blit(title, (10, 10))
    
    # Informações das vazões
    y_offset += 5
    flow_title = font.render("Vazões:", True, (200, 200, 255))
    screen.blit(flow_title, (10, y_offset))
    y_offset += 20
    
    for k, (i, j) in enumerate(edges):
        if k < len(q):
            flow_text = f"  {i}->{j}: q={q[k]:.2f}, R={R[k]:.2f}"
            text = font.render(flow_text, True, (200, 200, 200))
            screen.blit(text, (10, y_offset))
            y_offset += 18
    
    # Lista de teclas e funções
    y_offset += 10
    keys_title = font.render("Teclas:", True, (200, 200, 255))
    screen.blit(keys_title, (10, y_offset))
    y_offset += 20
    
    commands = [
        ("TAB", "Toggle debug"),
        ("N", "Nova rede aleatória"),
        ("T", "Adicionar ruído global"),
        ("Clique Esq", "Selecionar/Criar nó"),
        ("Clique Dir", "Criar aresta"),
        ("Shift+Clique", "Toggle pressão atm (nó)"),
        ("Arrastar", "Mover nó selecionado"),
        ("Q/A", "Aumentar/Diminuir Q (nó selecionado)"),
        ("R/F", "Aumentar/Diminuir R (aresta selecionada)"),
        ("DELETE", "Remover nó/aresta selecionada"),
        ("P", "Resetar partículas (nó selecionado)"),
    ]
    
    for key, desc in commands:
        text = font.render(f"{key}: {desc}", True, (180, 180, 180))
        screen.blit(text, (10, y_offset))
        y_offset += 18


def draw():
    screen.fill((20,20,30))
    rede = criar_rede(list(range(len(nodes))), edges, R, Q)
    q, p = resolver_rede(rede, ref_no=pressao_atm)

    # desenha arestas
    for k, (i, j) in enumerate(edges):
        x1, y1 = nodes[i]["pos"]
        x2, y2 = nodes[j]["pos"]

        color = (0,150,255) if k < len(q) and q[k] >= 0 else (255,100,100)
        if selected_edge == k:
            color = (255,255,0)

        Rmin, Rmax = 1.0, 10.0
        thick_min, thick_max = 1, 10
        val = 1 / (R[k] ** 0.25)
        vmin = 1 / (Rmax ** 0.25)
        vmax = 1 / (Rmin ** 0.25)
        t = max(0, min(1, (val - vmin) / (vmax - vmin)))
        thickness = int(thick_min + t * (thick_max - thick_min))

        pygame.draw.line(screen, color, (x1,y1), (x2,y2), thickness)

        if DEBUG:
            if k < len(q):
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                dx, dy = x2 - x1, y2 - y1
                L = max(1, (dx**2 + dy**2)**0.5)
                nx, ny = -dy / L, dx / L
                offset = 12
                tx, ty = int(mx + nx*offset), int(my + ny*offset)
                screen.blit(font.render(f"R={R[k]:.2f} q={q[k]:.2f}", True, (255,255,255)), (tx, ty))

    draw_flow_particles(q)

    for i, n in enumerate(nodes):
        x, y = n["pos"]
        color = (100,100,255)
        if selected_node == i:
            color = (255,255,0)
        pygame.draw.circle(screen, color, (x,y), 6)
        pressure = p[i] if i < len(p) else 0
        if DEBUG:
            screen.blit(font.render(f"{i}:Q={Q[i]:.1f} p={pressure:.2f}", True, (255,255,255)), (x+10,y))
        if i in pressao_atm:
            pygame.draw.circle(screen, (0,255,0), (x,y), 16,2)

    # Mostra linha "TAB:debug" quando debug está desativado
    if not DEBUG:
        debug_text = font.render("TAB:debug", True, (150, 150, 150))
        screen.blit(debug_text, (10, 10))
    else:
        # Mostra informações completas de debug
        draw_debug_info(q, p)

    if creating_edge is not None:
        pygame.draw.line(screen, (255,255,0), nodes[creating_edge]["pos"], pygame.mouse.get_pos(), 2)

    pygame.display.flip()

# ============================
# FUNÇÕES AUXILIARES
# ============================
def add_node(pos):
    nodes.append({"pos": pos})
    Q.append(0.0)

def remove_node(i):
    global selected_node, flow_particles, pressao_atm
    edges_to_remove = [k for k,(a,b) in enumerate(edges) if a==i or b==i]
    for k in reversed(edges_to_remove):
        edges.pop(k)
        R.pop(k)
    nodes.pop(i)
    Q.pop(i)
    # ajusta índices das arestas restantes
    for idx,(a,b) in enumerate(edges):
        edges[idx] = (a-(a>i), b-(b>i))
    pressao_atm = [r-(r>i) for r in pressao_atm if r!=i]
    if not pressao_atm and nodes:
        pressao_atm = [0]
    selected_node = None

    # Reconstrói flow_particles de acordo com as arestas atuais
    flow_particles = [[p/NUM_PARTICLES for p in range(NUM_PARTICLES)] for _ in edges]

def add_edge(i, j):
    edges.append((i,j))
    R.append(np.random.uniform(R_min,R_max))
    flow_particles.append([p/NUM_PARTICLES for p in range(NUM_PARTICLES)])

def remove_edge(k):
    edges.pop(k)
    R.pop(k)
    flow_particles.pop(k)

# ============================
# LOOP PRINCIPAL
# ============================
# gerar_rede_aleatoria()
gerar_rede_1()
running = True

while running:
    clock.tick(60)
    mouse_pos = np.array(pygame.mouse.get_pos())
    mods = pygame.key.get_mods()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # clique esquerdo
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                node = get_node_at(mouse_pos)
                edge = get_edge_at(mouse_pos)
                if node is not None:
                    if selected_node is not None:
                        selected_node = None
                    elif mods & pygame.KMOD_SHIFT:
                        if node not in pressao_atm:
                            pressao_atm.append(node)
                        else:
                            pressao_atm.remove(node)
                    else:
                        selected_node = node
                        selected_edge = None
                elif edge is not None:
                    if selected_edge is not None:
                        selected_edge = None
                    selected_edge = edge
                    selected_node = None
                else:
                    if selected_edge or selected_node:
                        selected_node = None
                        selected_edge = None
                    else:
                        add_node(tuple(mouse_pos))
                        selected_node = None
                        selected_edge = None

            # clique direito = criar aresta
            elif event.button == 3:
                node = get_node_at(mouse_pos)
                if node is not None:
                    if creating_edge is None:
                        creating_edge = node
                    else:
                        if node != creating_edge:
                            edge = tuple(sorted((creating_edge, node)))
                            if edge not in edges:
                                add_edge(*edge)
                        creating_edge = None

        if event.type == pygame.MOUSEMOTION:
            if selected_node is not None:
                nodes[selected_node]["pos"] = mouse_pos

        if event.type == pygame.KEYDOWN:
            # Toggle debug com TAB
            if event.key == pygame.K_TAB:
                DEBUG = not DEBUG
            
            # nós
            if selected_node is not None:
                if event.key == pygame.K_q:
                    Q[selected_node] += 1
                elif event.key == pygame.K_a:
                    Q[selected_node] -= 1
                elif event.key == pygame.K_DELETE:
                    remove_node(selected_node)
                    selected_node = None
                elif event.key == pygame.K_p:
                    # Reinicialização manual (opcional)
                    reset_particles()

            # arestas
            elif selected_edge is not None:
                if event.key == pygame.K_r:
                    R[selected_edge] += 0.5
                elif event.key == pygame.K_f:
                    R[selected_edge] = max(0.1, R[selected_edge] - 0.5)
                elif event.key == pygame.K_DELETE:
                    remove_edge(selected_edge)
                    selected_edge = None

            # teclas globais (sem seleção)
            else:
                # gerar rede aleatória
                if event.key == pygame.K_n:
                    gerar_rede_aleatoria()
                    selected_node = None
                    selected_edge = None
                    creating_edge = None
                # adicionar ruído global (tecla T, por exemplo)
                elif event.key == pygame.K_t:
                    alfa = 100.0
                    adicionar_ruido(R, alfa)

    # Antes de desenhar, sanitiza posições dos nós (evita nan)
    sanitize_nodes()
    draw()

pygame.quit()
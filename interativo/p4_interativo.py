import pygame
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import random

# Inicialização
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rede Hidráulica Interativa")
font = pygame.font.SysFont("Arial", 14)
font_small = pygame.font.SysFont("Arial", 11)
clock = pygame.time.Clock()

DEBUG = False

# Cores
COR = {
    'bg': (20,20,30), 'node': (100,100,255), 'node_atm': (0,255,0),
    'selected': (255,255,0), 'edge': (0,150,255), 'edge_reforcada': (0,255,200),
    'text': (255,255,255), 'dim': (150,150,150), 'panel': (40,40,50),
    'alert': (255,50,50), 'success': (0,200,0)
}

# Variáveis globais
selected_node = None
selected_edge = None
creating_edge = None

# Zoom e pan
zoom = 1.0
zoom_min, zoom_max = 0.3, 3.0
pan_x, pan_y = 0, 0
dragging = False
drag_start = None

# ============================
# MODELO MATEMÁTICO
# ============================
def criar_rede(vertices, arestas, R, Q):
    return {"V": vertices, "A": arestas, "R": np.array(R, dtype=float), "Q": np.array(Q, dtype=float)}

def construir_C(rede):
    n, m = len(rede["V"]), len(rede["A"])
    rows, cols, data = [], [], []
    for k, (i, j) in enumerate(rede["A"]):
        rows += [i, j]; cols += [k, k]; data += [1, -1]
    return sp.coo_matrix((data, (rows, cols)), shape=(n, m)).tocsr()

def construir_D(rede):
    n, m = len(rede["V"]), len(rede["A"])
    R = rede["R"]
    rows, cols, data = [], [], []
    for k, (i, j) in enumerate(rede["A"]):
        val = 1.0 / max(R[k], 0.001)
        rows += [k, k]; cols += [i, j]; data += [-val, val]
    return sp.coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()

def montar_Z(C, D, ref_no):
    n, m = C.shape
    if isinstance(ref_no, int): ref_no = [ref_no]
    C_mod = C.tolil()
    for r in ref_no:
        if r < n: C_mod[r, :] = 0
    C_mod = C_mod.tocsr()
    F = sp.lil_matrix((n, n))
    for r in ref_no:
        if r < n: F[r, r] = 1.0
    F = F.tocsr()
    I = sp.eye(m, format='csr')
    return sp.bmat([[C_mod, F], [I, D]], format='csr')

def montar_b(Q, m, ref_no):
    n = len(Q)
    if isinstance(ref_no, int): ref_no = [ref_no]
    b = np.zeros(n + m)
    b[:n] = Q
    for r in ref_no:
        if r < n: b[r] = 0.0
    return b

def resolver_rede(rede, ref_no):
    if len(rede["A"]) == 0:
        return np.array([]), np.zeros(len(rede["V"]))
    C = construir_C(rede)
    D = construir_D(rede)
    Z = montar_Z(C, D, ref_no)
    b = montar_b(rede["Q"], len(rede["A"]), ref_no)
    try:
        x = spla.spsolve(Z, b)
        return x[:len(rede["A"])], x[len(rede["A"]):]
    except:
        return np.zeros(len(rede["A"])), np.zeros(len(rede["V"]))

# ============================
# REDES PRÉ-DEFINIDAS
# ============================
def rede_media():
    vertices = list(range(8))
    arestas = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,5),(2,6),(3,6),(3,7),(4,5),(5,6),(6,7)]
    R = [2.0,1.5,2.5,1.8,2.2,1.6,2.0,2.3,1.9,2.1,1.7,2.4]
    Q = [15,0,0,0,0,0,0,0]
    nos_pressao_atm = [5,7]
    return vertices, arestas, R, Q, nos_pressao_atm

def rede_simples():
    vertices = [0,1,2,3]
    arestas = [(0,1),(0,2),(1,2),(1,3)]
    R = [2,3,1,2]
    Q = [12,0,0,0]
    nos_pressao_atm = [3]
    return vertices, arestas, R, Q, nos_pressao_atm

def gerar_rede_aleatoria():
    global nodes, edges, R, Q, pressao_atm, flow_particles, vertices
    num_nos = random.randint(5, 10)
    nodes = [{"pos": (random.randint(100, WIDTH-100), random.randint(100, HEIGHT-100))} for _ in range(num_nos)]
    nos = list(range(num_nos))
    arestas = set()
    for i in range(1, num_nos):
        arestas.add((random.randint(0, i-1), i))
    max_arestas = num_nos * (num_nos - 1) // 2
    num_arestas = min(random.randint(6, 12), max_arestas)
    while len(arestas) < num_arestas:
        i, j = random.sample(nos, 2)
        arestas.add(tuple(sorted((i, j))))
    edges = list(arestas)
    R = [random.uniform(1.0, 5.0) for _ in edges]
    Q = [0]*num_nos
    Q[random.choice(nos)] = random.uniform(5, 20)
    pressao_atm = [random.choice([n for n in nos if Q[n]==0])]
    vertices = list(range(num_nos))
    flow_particles = [[p/5 for p in range(5)] for _ in edges]

# ============================
# FUNÇÕES DE TRANSFORMAÇÃO (ZOOM/PAN)
# ============================
def world_to_screen(pos):
    x = pos[0] * zoom + WIDTH//2 + pan_x
    y = pos[1] * zoom + HEIGHT//2 + pan_y
    return (x, y)

def screen_to_world(pos):
    x = (pos[0] - WIDTH//2 - pan_x) / zoom
    y = (pos[1] - HEIGHT//2 - pan_y) / zoom
    return (x, y)

def aplicar_zoom(delta, mouse_pos):
    global zoom, pan_x, pan_y
    mx = (mouse_pos[0] - WIDTH//2 - pan_x) / zoom
    my = (mouse_pos[1] - HEIGHT//2 - pan_y) / zoom
    nz = max(zoom_min, min(zoom_max, zoom * (1 + delta*0.1)))
    if nz != zoom:
        pan_x += mx * zoom - mx * nz
        pan_y += my * zoom - my * nz
        zoom = nz

# ============================
# DETEÇÃO DE NÓ E ARESTA
# ============================
def get_node_at(pos):
    for i, n in enumerate(nodes):
        sx, sy = world_to_screen(n["pos"])
        if np.linalg.norm(np.array([sx, sy]) - pos) < 12/zoom:
            return i
    return None

def get_edge_at(pos):
    th = 6/zoom
    for k, (i, j) in enumerate(edges):
        p1 = world_to_screen(nodes[i]["pos"])
        p2 = world_to_screen(nodes[j]["pos"])
        p1, p2 = np.array(p1), np.array(p2)
        v = p2 - p1
        L = np.linalg.norm(v)
        if L == 0: continue
        proj = np.dot(pos - p1, v / L)
        if 0 <= proj <= L:
            if np.linalg.norm(p1 + v/L*proj - pos) < th:
                return k
    return None

# ============================
# DESENHO
# ============================
NUM_PARTICLES = 5
flow_particles = []

def draw_flow_particles(q):
    for k, (i, j) in enumerate(edges):
        if k >= len(q) or np.isnan(q[k]) or abs(q[k]) < 0.01:
            continue
        p1 = np.array(world_to_screen(nodes[i]["pos"]))
        p2 = np.array(world_to_screen(nodes[j]["pos"]))
        d = p2 - p1
        L = np.linalg.norm(d)
        if L == 0: continue
        d /= L
        speed = q[k] * 0.002
        for idx in range(len(flow_particles[k])):
            flow_particles[k][idx] = (flow_particles[k][idx] + speed) % 1.0
            pos = p1 + flow_particles[k][idx] * d * L
            cor = (0,255,255) if q[k] >= 0 else (255,100,100)
            pygame.draw.circle(screen, cor, pos.astype(int), max(2, int(4/zoom)))

def draw():
    global selected_node, selected_edge, creating_edge
    screen.fill(COR['bg'])
    rede = criar_rede(vertices, edges, R, Q)
    q, p = resolver_rede(rede, pressao_atm)
    # Lista com pressões dos nós que NÃO estão em pressao_atm
    lista = [p[i] for i in range(len(p)) if i not in pressao_atm]
    p_min = min(lista)
    p_max = max(lista)

    # Desenha arestas
    for k, (i, j) in enumerate(edges):
        p1 = world_to_screen(nodes[i]["pos"])
        p2 = world_to_screen(nodes[j]["pos"])
        cor = COR['edge']
        if k == selected_edge:
            cor = COR['selected']
        esp = max(1, int(8/(max(R[k],0.1)**0.3)))
        pygame.draw.line(screen, cor, p1, p2, esp)
        if DEBUG and k < len(q):
            mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
            t = font_small.render(f"R={R[k]:.2f} q={q[k]:.2f}", True, COR['dim'])
            screen.blit(t, (mx-30, my-10))

    draw_flow_particles(q)

    # Desenha nós
    for i, n in enumerate(nodes):
        x, y = world_to_screen(n["pos"])
        cor = COR['node_atm'] if i in pressao_atm else COR['node']
        if i == selected_node:
            cor = COR['selected']
        r = max(5, int(10/zoom))
        pygame.draw.circle(screen, cor, (int(x), int(y)), r)
        pygame.draw.circle(screen, COR['text'], (int(x), int(y)), r, max(1, int(2/zoom)))
        if DEBUG:
            press = p[i] if i < len(p) else 0
            t = font_small.render(f"{i}:Q={Q[i]:.1f} p={press:.1f}", True, COR['text'])
            screen.blit(t, (x+10, y-10))
        elif zoom > 0.5:
            t = font_small.render(f"{i}", True, COR['text'])
            screen.blit(t, (x+8, y-8))

    if creating_edge is not None:
        x, y = world_to_screen(nodes[creating_edge]["pos"])
        pygame.draw.line(screen, COR['selected'], (x, y), pygame.mouse.get_pos(), 2)

    # Painel
    panel = pygame.Rect(10, 10, 260, 600)
    pygame.draw.rect(screen, COR['panel'], panel, border_radius=5)
    pygame.draw.rect(screen, COR['dim'], panel, 2, border_radius=5)
    y = 20
    titulo = font.render("CONTROLES", True, COR['text'])
    screen.blit(titulo, (20, y)); y += 25
    text_p_min = font.render(f"p_min:{p_min:.4f}",True,COR['text'])
    screen.blit(text_p_min, (20, y)); y += 25
    text_p_max = font.render(f"p_max:{p_max:.4f}",True,COR['text'])
    screen.blit(text_p_max, (20, y)); y += 25
    text_p = font.render(f"diff:{p_max-p_min:.4f}",True,COR['text'])
    screen.blit(text_p, (20, y)); y += 25
    y+=25

    txts = ["Clique: selecionar", "Shift+clique: ATM", "Clique dir: aresta",
            "Arrastar: mover nó", "Q/A: +/- Q", "R/F: +/- R", "Delete: remover",
            "Scroll: zoom", "Botão médio: pan", "N: reset", "E: Entupimento", "O: Otimizar"]
    for txt in txts:
        t = font_small.render(txt, True, COR['dim'])
        screen.blit(t, (20, y)); y += 18

    # Mostra zoom
    t = font_small.render(f"Zoom: {zoom:.1f}x", True, COR['dim'])
    screen.blit(t, (WIDTH-100, HEIGHT-30))

    pygame.display.flip()

# ============================
# FUNÇÕES AUXILIARES
# ============================
def add_node(pos):
    vertices.append(len(nodes))
    nodes.append({"pos": pos})
    Q.append(0.0)
    return len(nodes)-1

def remove_node(i):
    global selected_node, flow_particles, pressao_atm, vertices
    rm_edges = [k for k,(a,b) in enumerate(edges) if a==i or b==i]
    for k in sorted(rm_edges, reverse=True):
        edges.pop(k); R.pop(k); flow_particles.pop(k)
    nodes.pop(i); Q.pop(i); vertices.pop(i)
    pressao_atm = [r-(r>i) for r in pressao_atm if r!=i]
    for idx, (a,b) in enumerate(edges):
        edges[idx] = (a-(a>i), b-(b>i))
    selected_node = None
    flow_particles = [[p/NUM_PARTICLES for p in range(NUM_PARTICLES)] for _ in edges]

def add_edge(i, j):
    edges.append((i,j))
    R.append(random.uniform(1.0, 5.0))
    flow_particles.append([p/NUM_PARTICLES for p in range(NUM_PARTICLES)])

def remove_edge(k):
    edges.pop(k); R.pop(k); flow_particles.pop(k)

def testar_entupimento():
    global R
    R_orig = R.copy()
    estouros = 0
    total = 50
    for _ in range(total):
        R = R_orig.copy()
        for k in range(len(edges)):
            if random.random() < 0.1:
                R[k] *= 100
        rede = criar_rede(vertices, edges, R, Q)
        _, p = resolver_rede(rede, pressao_atm)
        masc = [i for i in vertices if i not in pressao_atm]
        if masc and max(p[masc]) > 200:
            estouros += 1
    R = R_orig.copy()
    prob = estouros/total
    print(f"Probabilidade de estouro: {100*prob:.1f}%")
    return prob

def otimizar_reforcos(T):
    global R, edges
    if T > len(edges): return []
    m = len(edges)
    masc = [i for i in vertices if i not in pressao_atm]
    _, p_orig = resolver_rede(criar_rede(vertices, edges, R, Q), pressao_atm)
    dif_orig = max(p_orig[masc]) - min(p_orig[masc]) if masc else 1
    
    def fitness(x):
        R_mod = R.copy()
        for k in range(m):
            if x[k]: R_mod[k] = 0.01
        _, p = resolver_rede(criar_rede(vertices, edges, R_mod, Q), pressao_atm)
        pv = [p[i] for i in masc if i < len(p)]
        dif = max(pv) - min(pv) if pv else dif_orig
        return (dif_orig - dif) / dif_orig if dif_orig else 0
    
    pop = [np.random.rand(m) for _ in range(30)]
    melhor_fit, melhor_x = -1, None
    for _ in range(30):
        fits = []
        for c in pop:
            x = np.zeros(m); x[np.argsort(c)[-T:]] = 1
            fits.append(fitness(x))
        idx = np.argsort(fits)[::-1]
        pop = [pop[i] for i in idx]
        if fits[idx[0]] > melhor_fit:
            melhor_fit = fits[idx[0]]
            melhor_x = np.zeros(m); melhor_x[np.argsort(pop[0])[-T:]] = 1
        elite = pop[:6]
        nova = elite[:]
        while len(nova) < 30:
            p1, p2 = random.choice(elite), random.choice(pop[6:])
            filho = p1 * 0.7 + p2 * 0.3
            if random.random() < 0.3:
                filho += np.random.normal(0, 0.1, m)
            nova.append(np.clip(filho, 0, 1))
        pop = nova
    return [k for k in range(m) if melhor_x[k]]

# ============================
# INICIALIZAÇÃO
# ============================
vertices, edges, R, Q, pressao_atm = rede_media()
nodes = [{"pos": ((random.random()-0.5)*2, (random.random()-0.5)*2)} for _ in vertices]
flow_particles = [[p/NUM_PARTICLES for p in range(NUM_PARTICLES)] for _ in edges]

# ============================
# LOOP PRINCIPAL
# ============================
running = True
while running:
    
    clock.tick(60)
    mouse_pos = pygame.mouse.get_pos()
    mods = pygame.key.get_mods()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Esquerdo
                node = get_node_at(mouse_pos)
                edge = get_edge_at(mouse_pos)
                if node is not None:
                    if selected_node is not None:
                        selected_node = None
                    elif mods & pygame.KMOD_SHIFT:
                        if node in pressao_atm:
                            pressao_atm.remove(node)
                        else:
                            pressao_atm.append(node)
                    else:
                        selected_node = node
                        selected_edge = None
                elif edge is not None:
                    selected_edge = edge if selected_edge != edge else None
                    selected_node = None
                else:
                    # Cria novo nó na posição do clique
                    selected_node = None
                    selected_edge = None
                    world_pos = screen_to_world(mouse_pos)
                    add_node(world_pos)
                    
            elif event.button == 2:  # Médio - pan
                dragging = True
                drag_start = mouse_pos
                
            elif event.button == 3:  # Direito - criar aresta
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
                        
            elif event.button == 4:  # Scroll up
                aplicar_zoom(0.1, mouse_pos)
            elif event.button == 5:  # Scroll down
                aplicar_zoom(-0.1, mouse_pos)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 2:
                dragging = False
                drag_start = None

        elif event.type == pygame.MOUSEMOTION:
            if selected_node is not None:
                world_pos = screen_to_world(mouse_pos)
                nodes[selected_node]["pos"] = world_pos
            elif dragging:
                dx = mouse_pos[0] - drag_start[0]
                dy = mouse_pos[1] - drag_start[1]
                pan_x += dx
                pan_y += dy
                drag_start = mouse_pos

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                DEBUG = not DEBUG
            elif event.key == pygame.K_n:
                # global zoom, pan_x, pan_y
                zoom, pan_x, pan_y = 1.0, 0, 0
            elif event.key == pygame.K_e:
                prob = testar_entupimento()
                # Mostra resultado na tela
                draw()
                overlay = pygame.Surface((WIDTH, HEIGHT))
                overlay.set_alpha(200)
                overlay.fill(COR['bg'])
                screen.blit(overlay, (0,0))
                box = pygame.Rect(WIDTH//2-250, HEIGHT//2-100, 500, 150)
                pygame.draw.rect(screen, COR['panel'], box, border_radius=10)
                t = font.render(f"Estouro: {100*prob:.1f}%", True, COR['text'])
                screen.blit(t, (box.centerx-t.get_width()//2, box.centery-20))
                pygame.display.flip()
                pygame.time.wait(2000)
            elif event.key == pygame.K_o:
                T = min(1, len(edges))
                reforcadas = otimizar_reforcos(T)
                for k in reforcadas:
                    if k < len(R):
                        R[k] = 0.01
                print(f"Arestas reforçadas: {reforcadas}")
            
            elif selected_node is not None:
                if event.key == pygame.K_q:
                    Q[selected_node] += 1
                elif event.key == pygame.K_a:
                    Q[selected_node] = max(0, Q[selected_node] - 1)
                elif event.key == pygame.K_DELETE:
                    remove_node(selected_node)
                    selected_node = None
                    
            elif selected_edge is not None:
                if event.key == pygame.K_r:
                    R[selected_edge] += 0.5
                elif event.key == pygame.K_f:
                    R[selected_edge] = max(0.01, R[selected_edge] - 0.5)
                elif event.key == pygame.K_DELETE:
                    remove_edge(selected_edge)
                    selected_edge = None

    draw()

pygame.quit()
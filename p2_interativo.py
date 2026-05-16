import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pygame
import random
from collections import defaultdict

# Inicializa PyGame
pygame.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Análise de Entupimentos - Visualização da Rede")
font = pygame.font.SysFont("Arial", 16)
font_small = pygame.font.SysFont("Arial", 12)
font_title = pygame.font.SysFont("Arial", 18, bold=True)
clock = pygame.time.Clock()

# Cores
COLORS = {
    'background': (20, 20, 30),
    'node': (100, 100, 255),
    'node_atm': (0, 255, 0),
    'edge': (0, 150, 255),
    'edge_clogged': (255, 100, 0),
    'text': (255, 255, 255),
    'text_dim': (150, 150, 150),
    'panel': (40, 40, 50),
    'alert': (255, 50, 50),
    'success': (0, 200, 0),
    'button': (60, 60, 80),
    'button_hover': (80, 80, 100)
}

# ============================
# FUNÇÕES DE LAYOUT DO GRAFO
# ============================

def hierarchical_layout(vertices, arestas, width=WIDTH, height=HEIGHT):
    """Layout hierárquico simples baseado em níveis"""
    n = len(vertices)
    if n == 0:
        return []
    
    # Calcula níveis (profundidade) aproximada
    adj = defaultdict(list)
    for i, j in arestas:
        adj[i].append(j)
        adj[j].append(i)
    
    # BFS para determinar níveis
    levels = [-1] * n
    if n > 0:
        queue = [0]
        levels[0] = 0
        
        while queue:
            node = queue.pop(0)
            for neighbor in adj[node]:
                if levels[neighbor] == -1:
                    levels[neighbor] = levels[node] + 1
                    queue.append(neighbor)
    
    # Se algum nó não foi alcançado, atribui nível 0
    for i in range(n):
        if levels[i] == -1:
            levels[i] = 0
    
    max_level = max(levels) if levels else 0
    
    # Distribui os nós
    positions = []
    level_counts = defaultdict(int)
    for i in range(n):
        level_counts[levels[i]] += 1
    
    y_spacing = height / (max_level + 2)
    x_spacing = width / (max(level_counts.values()) + 2) if level_counts else width / 2
    
    level_positions = defaultdict(int)
    
    for i in range(n):
        level = levels[i]
        x = (level_positions[level] + 1) * x_spacing
        y = (level + 1) * y_spacing
        positions.append((x, y))
        level_positions[level] += 1
    
    return positions

# ============================
# FUNÇÕES DE REDE
# ============================

def criar_rede(vertices, arestas, R, Q):
    return {
        "V": vertices,
        "A": arestas,
        "R": np.array(R, dtype=float),
        "Q": np.array(Q, dtype=float)
    }

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
    n, m = C.shape
    
    if isinstance(ref_nos, int):
        ref_nos = [ref_nos]
    
    C_mod = C.tolil()
    for no in ref_nos:
        C_mod[no, :] = 0
    C_mod = C_mod.tocsr()
    
    F = sp.lil_matrix((n, n))
    for no in ref_nos:
        F[no, no] = 1.0
    F = F.tocsr()
    
    I = sp.eye(m, format='csr')
    return sp.bmat([[C_mod, F], [I, D]], format='csr')

def montar_b_com_condicao(Q, m, ref_nos):
    if isinstance(ref_nos, int):
        ref_nos = [ref_nos]
    
    b = np.zeros(len(Q) + m)
    b[:len(Q)] = Q
    
    for no in ref_nos:
        b[no] = 0.0
    
    return b

def resolver_rede(rede, ref_nos):
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
        return np.zeros(m), np.zeros(len(rede["V"]))

def adicionar_ruido(R, len_edges, alfa, chance_mudanca=0.1):
    R_mod = R.copy()
    clogged_edges = []
    
    for k in range(len_edges):
        if random.uniform(0, 1) < chance_mudanca:
            R_mod[k] *= alfa
            clogged_edges.append(k)
    
    return R_mod, clogged_edges

def rede_simples():
    vertices = [0, 1, 2, 3]
    arestas = [(0, 1), (0, 2), (1, 2), (1, 3)]
    nos_pressao_atm = [3]
    R = [2, 3, 1, 2]
    Q = [12, 0, 0, 0]
    return vertices, arestas, R, Q, nos_pressao_atm

def rede_media():
    vertices = [0, 1, 2, 3, 4, 5, 6, 7]
    arestas = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 5), (2, 6),
        (3, 6), (3, 7),
        (4, 5), (5, 6), (6, 7)
    ]
    nos_pressao_atm = [5, 7]
    R = [2.0, 1.5, 2.5, 1.8, 2.2, 1.6, 2.0, 2.3, 1.9, 2.1, 1.7, 2.4]
    Q = [15, 0, 0, 0, 0, 0, 0, 0]
    return vertices, arestas, R, Q, nos_pressao_atm

# ============================
# TELA DE MENU
# ============================

class Botao:
    def __init__(self, x, y, w, h, texto, cor_normal, cor_hover, text_color=COLORS['text']):
        self.rect = pygame.Rect(x, y, w, h)
        self.texto = texto
        self.cor_normal = cor_normal
        self.cor_hover = cor_hover
        self.text_color = text_color
        self.hover = False
    
    def desenhar(self, screen):
        cor = self.cor_hover if self.hover else self.cor_normal
        pygame.draw.rect(screen, cor, self.rect, border_radius=5)
        pygame.draw.rect(screen, COLORS['text_dim'], self.rect, 2, border_radius=5)
        
        text_surface = font.render(self.texto, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
    
    def atualizar(self, mouse_pos):
        self.hover = self.rect.collidepoint(mouse_pos)
        return self.hover
    
    def clicado(self, mouse_pos, mouse_clicado):
        return self.rect.collidepoint(mouse_pos) and mouse_clicado

class ConfiguracaoItem:
    def __init__(self, x, y, label, valor_min, valor_max, valor_inicial, formato="{:.0f}", step=1):
        self.x = x
        self.y = y
        self.label = label
        self.valor = valor_inicial
        self.valor_min = valor_min
        self.valor_max = valor_max
        self.formato = formato
        self.step = step
        
        # Botões
        self.btn_menos = Botao(x + 150, y - 5, 30, 30, "-", COLORS['button'], COLORS['button_hover'])
        self.btn_mais = Botao(x + 190, y - 5, 30, 30, "+", COLORS['button'], COLORS['button_hover'])
    
    def desenhar(self, screen):
        # Label
        text = font.render(self.label, True, COLORS['text'])
        screen.blit(text, (self.x, self.y))
        
        # Valor
        valor_text = font.render(self.formato.format(self.valor), True, COLORS['text_dim'])
        screen.blit(valor_text, (self.x + 100, self.y))
        
        # Botões
        self.btn_menos.desenhar(screen)
        self.btn_mais.desenhar(screen)
    
    def atualizar(self, mouse_pos, mouse_clicado):
        self.btn_menos.atualizar(mouse_pos)
        self.btn_mais.atualizar(mouse_pos)
        
        if mouse_clicado:
            if self.btn_menos.clicado(mouse_pos, mouse_clicado):
                self.valor = max(self.valor_min, self.valor - self.step)
            elif self.btn_mais.clicado(mouse_pos, mouse_clicado):
                self.valor = min(self.valor_max, self.valor + self.step)
        
        return self.valor

def tela_menu():
    """Tela inicial com opções"""
    titulo = font_title.render("ANÁLISE DE ENTUPIMENTOS EM REDES HIDRÁULICAS", True, COLORS['text'])
    y_offset = -200
    # Botões principais
    btn_simples = Botao(WIDTH//2 - 150,y_offset+ HEIGHT - 150, 300, 50, 
                        "REDE SIMPLES (4 nós)", COLORS['button'], COLORS['button_hover'])
    btn_media = Botao(WIDTH//2 - 150, y_offset+HEIGHT - 90, 300, 50, 
                      "REDE MÉDIA (8 nós)", COLORS['button'], COLORS['button_hover'])
    btn_sair = Botao(WIDTH//2 - 150,y_offset+ HEIGHT - 30, 300, 50, 
                     "SAIR", COLORS['button'], (100, 50, 50))
    
    # Configurações com botões + e -
    configs = {
        'ensaios': ConfiguracaoItem(WIDTH//2 - 200, 200, "Ensaios:", 1, 50, 5, step=1),
        'alfa': ConfiguracaoItem(WIDTH//2 - 200, 250, "Fator (α):", 1, 10000, 1000, formato="{:.0f}", step=100),
        'chance': ConfiguracaoItem(WIDTH//2 - 200, 300, "Chance (r):", 0.01, 0.5, 0.1, formato="{:.2f}", step=0.01),
        'p_limite': ConfiguracaoItem(WIDTH//2 - 200, 350, "P limite:", 50, 500, 200, step=10)
    }
    
    running_menu = True
    
    while running_menu:
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicado = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicado = True
        
        # Atualiza botões principais
        btn_simples.atualizar(mouse_pos)
        btn_media.atualizar(mouse_pos)
        btn_sair.atualizar(mouse_pos)
        
        # Atualiza configurações
        for config in configs.values():
            config.atualizar(mouse_pos, mouse_clicado)
        
        # Verifica cliques nos botões principais
        if btn_simples.clicado(mouse_pos, mouse_clicado):
            return "simples", (configs['ensaios'].valor, configs['alfa'].valor, 
                              configs['chance'].valor, configs['p_limite'].valor)
        if btn_media.clicado(mouse_pos, mouse_clicado):
            return "media", (configs['ensaios'].valor, configs['alfa'].valor, 
                            configs['chance'].valor, configs['p_limite'].valor)
        if btn_sair.clicado(mouse_pos, mouse_clicado):
            return None, None
        
        # Desenha
        screen.fill(COLORS['background'])
        
        # Título
        titulo_rect = titulo.get_rect(center=(WIDTH//2, 80))
        screen.blit(titulo, titulo_rect)
        
        # Painel de configurações
        config_panel = pygame.Rect(WIDTH//2 - 250, 140, 500, 260)
        pygame.draw.rect(screen, COLORS['panel'], config_panel, border_radius=10)
        pygame.draw.rect(screen, COLORS['text_dim'], config_panel, 2, border_radius=10)
        
        # Título do painel
        panel_title = font_title.render("CONFIGURAÇÕES", True, COLORS['text'])
        screen.blit(panel_title, (config_panel.centerx - panel_title.get_width()//2, config_panel.y + 10))
        
        # Desenha configurações
        for config in configs.values():
            config.desenhar(screen)
        
        # Botões principais
        btn_simples.desenhar(screen)
        btn_media.desenhar(screen)
        btn_sair.desenhar(screen)
        
        # Instruções
        instr = font_small.render("Use os botões + e - para ajustar as configurações", True, COLORS['text_dim'])
        screen.blit(instr, (WIDTH//2 - 200, y_offset+HEIGHT - 180))
        
        pygame.display.flip()
        clock.tick(60)
    
    return None, None

# ============================
# SIMULAÇÃO VISUAL
# ============================

class SimulacaoVisual:
    def __init__(self, rede_func, nome_rede, qtd_ensaios=10, p_max_limite=200, 
                 alfa=10e2, chance_entupimento=0.1):
        
        self.rede_func = rede_func
        self.nome_rede = nome_rede
        self.qtd_ensaios = qtd_ensaios
        self.p_max_limite = p_max_limite
        self.alfa = alfa
        self.chance_entupimento = chance_entupimento
        
        # Carrega rede base
        self.vertices, self.arestas, self.R_base, self.Q, self.nos_pressao_atm = rede_func()
        
        # Calcula layout
        self.positions = hierarchical_layout(self.vertices, self.arestas)
        
        # Estado da simulação
        self.simulando = True
        self.pausado = False
        self.ensaio_atual = 0
        self.simulacao_atual = 0
        self.historico = []
        self.resultados_mostrados = False
        
        # Dados atuais
        self.R_atual = self.R_base.copy()
        self.q_atual = None
        self.p_atual = None
        self.clogged_edges = []
        self.estourou = False
        
        # Configura simulações
        self.total_simulacoes_por_ensaio = 30 + 10 * int(np.sqrt(len(self.arestas) * len(self.vertices)))
        self.total_simulacoes = self.qtd_ensaios * self.total_simulacoes_por_ensaio
        self.ensaio_estouros = 0
        
        # Inicializa
        self.atualizar_rede()
    
    def atualizar_rede(self):
        """Aplica entupimento e resolve a rede atual"""
        self.R_atual, self.clogged_edges = adicionar_ruido(
            self.R_base, len(self.arestas), 
            alfa=self.alfa, 
            chance_mudanca=self.chance_entupimento
        )
        
        rede = criar_rede(self.vertices, self.arestas, self.R_atual, self.Q)
        self.q_atual, self.p_atual = resolver_rede(rede, self.nos_pressao_atm)
        
        # Verifica estouro
        max_press = np.max(self.p_atual) if len(self.p_atual) > 0 else 0
        self.estourou = max_press > self.p_max_limite
        
        if self.estourou:
            self.ensaio_estouros += 1
    
    def proxima_simulacao(self):
        """Avança para a próxima simulação"""
        self.simulacao_atual += 1
        
        if self.simulacao_atual >= self.total_simulacoes_por_ensaio:
            # Fim do ensaio
            prob = self.ensaio_estouros / self.total_simulacoes_por_ensaio
            self.historico.append(prob)
            
            self.ensaio_atual += 1
            self.simulacao_atual = 0
            self.ensaio_estouros = 0
            
            if self.ensaio_atual >= self.qtd_ensaios:
                self.simulando = False
                return False
        else:
            # Continua no mesmo ensaio
            pass
        
        # Atualiza para a próxima rede
        self.atualizar_rede()
        return True
    
    def desenhar(self, screen):
        screen.fill(COLORS['background'])
        
        # Desenha arestas
        for k, (i, j) in enumerate(self.arestas):
            if i >= len(self.positions) or j >= len(self.positions):
                continue
            
            x1, y1 = self.positions[i]
            x2, y2 = self.positions[j]
            
            # Cor da aresta
            if k in self.clogged_edges:
                color = COLORS['edge_clogged']
            elif self.q_atual is not None and k < len(self.q_atual):
                if self.q_atual[k] >= 0:
                    intensity = min(200, int(abs(self.q_atual[k]) * 8))
                    color = (0, 100 + intensity//3, 200)
                else:
                    intensity = min(200, int(abs(self.q_atual[k]) * 8))
                    color = (200, 100 + intensity//3, 100)
            else:
                color = COLORS['edge']
            
            # Espessura
            thickness = 2
            if self.R_atual and k < len(self.R_atual):
                thickness = max(1, min(5, int(12 / (self.R_atual[k] ** 0.3))))
            
            pygame.draw.line(screen, color, (int(x1), int(y1)), (int(x2), int(y2)), thickness)
            
            # Mostra vazão
            if self.q_atual is not None and k < len(self.q_atual) and abs(self.q_atual[k]) > 0.1:
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                text = font_small.render(f"{abs(self.q_atual[k]):.1f}", True, COLORS['text_dim'])
                screen.blit(text, (mx - 12, my - 8))
        
        # Desenha nós
        for i, pos in enumerate(self.positions):
            x, y = pos
            
            # Cor do nó
            if i in self.nos_pressao_atm:
                color = COLORS['node_atm']
            else:
                color = COLORS['node']
            
            # Tamanho
            radius = 12
            if self.p_atual is not None and i < len(self.p_atual):
                radius = 8 + min(10, int(abs(self.p_atual[i]) / 15))
            
            pygame.draw.circle(screen, color, (int(x), int(y)), radius)
            pygame.draw.circle(screen, COLORS['text'], (int(x), int(y)), radius, 2)
            
            # Mostra pressão
            if self.p_atual is not None and i < len(self.p_atual):
                press_text = font_small.render(f"{self.p_atual[i]:.1f}", True, COLORS['text'])
                screen.blit(press_text, (x + 12, y - 8))
        
        # Painel de informações
        panel_rect = pygame.Rect(10, 10, 380, 240)
        # pygame.draw.rect(screen, COLORS['panel'], panel_rect)
        # pygame.draw.rect(screen, COLORS['text_dim'], panel_rect, 2)
        
        y = 20
        titulo = font_title.render(self.nome_rede, True, COLORS['text'])
        screen.blit(titulo, (20, y))
        y += 30
        
        
        # Progresso
        texto = f"Ensaio: {self.ensaio_atual + 1}/{self.qtd_ensaios}"
        prog_text = font.render(texto, True, COLORS['text'])
        screen.blit(prog_text, (20, y))
        y += 22
        
        total_feito = self.ensaio_atual * self.total_simulacoes_por_ensaio + self.simulacao_atual
        texto = f"Progresso: {total_feito}/{self.total_simulacoes}"
        prog_text = font.render(texto, True, COLORS['text'])
        screen.blit(prog_text, (20, y))
        y += 22
        
        # Configurações atuais
        config_text = font_small.render(f"α={self.alfa:.0f} | r={self.chance_entupimento:.2f} | Pmax={self.p_max_limite}", True, COLORS['text_dim'])
        screen.blit(config_text, (20, y))
        y += 22
        
        # Pressões
        if self.p_atual is not None and len(self.p_atual) > 0:
            max_p = np.max(self.p_atual)
            min_p = np.min(self.p_atual)
            press_text = font.render(f"Pmax: {max_p:.1f} | Pmin: {min_p:.1f}", True, COLORS['text'])
            screen.blit(press_text, (20, y))
            y += 22
            
            if self.estourou:
                alert = font.render(f"⚠️ ESTOURO! P > {self.p_max_limite}", True, COLORS['alert'])
                screen.blit(alert, (20, y))
                y += 22
        
        # Estatísticas do ensaio
        if self.simulacao_atual > 0:
            prob_atual = self.ensaio_estouros / self.simulacao_atual
            texto = f"Estouros: {self.ensaio_estouros}/{self.simulacao_atual} ({100*prob_atual:.1f}%)"
            est_text = font_small.render(texto, True, COLORS['text_dim'])
            screen.blit(est_text, (20, y))
        
        # Botão de pausa na tela
        btn_pause = pygame.Rect(WIDTH - 120, 20, 100, 30)
        cor_pause = COLORS['button_hover'] if self.pausado else COLORS['button']
        pygame.draw.rect(screen, cor_pause, btn_pause, border_radius=5)
        text_pause = "PAUSADO" if self.pausado else "PAUSAR"
        pause_text = font.render(text_pause, True, COLORS['text'])
        screen.blit(pause_text, (btn_pause.x + 15, btn_pause.y + 7))
        
        # Legenda
        legend_y = HEIGHT - 80
        legend_items = [
            ("Nó normal", COLORS['node']),
            ("Nó ATM", COLORS['node_atm']),
            ("Normal", COLORS['edge']),
            ("Entupido", COLORS['edge_clogged'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x = 20 + i * 130
            pygame.draw.circle(screen, color, (x, legend_y), 6)
            text = font_small.render(label, True, COLORS['text'])
            screen.blit(text, (x + 12, legend_y - 5))
        
        # Instruções
        instr_y = HEIGHT - 30
        instr = font_small.render("ESC: Voltar | ESPACO: Pausar", True, COLORS['text_dim'])
        screen.blit(instr, (WIDTH - 300, instr_y))
        
        return btn_pause
    
    def mostrar_resultados(self):
        """Exibe os resultados finais na tela"""
        if len(self.historico) == 0:
            return
        
        media = np.mean(self.historico)
        desvio = np.std(self.historico)
        
        # Mostra resultados na tela
        esperando = True
        while esperando:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE:
                        esperando = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    esperando = False
            
            self.desenhar(screen)
            
            # Sobreposição de resultados
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(200)
            overlay.fill(COLORS['background'])
            screen.blit(overlay, (0, 0))
            
            # Caixa de resultados
            result_box = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 - 100, 500, 200)
            pygame.draw.rect(screen, COLORS['panel'], result_box, border_radius=10)
            pygame.draw.rect(screen, COLORS['text_dim'], result_box, 3, border_radius=10)
            
            titulo = font_title.render("RESULTADOS FINAIS", True, COLORS['text'])
            screen.blit(titulo, (result_box.centerx - titulo.get_width()//2, result_box.y + 20))
            
            result_text = font.render(f"Probabilidade de estouro: {100*media:.2f}%", True, COLORS['text'])
            screen.blit(result_text, (result_box.centerx - result_text.get_width()//2, result_box.y + 70))
            
            desv_text = font.render(f"Desvio padrão: ±{100*desvio:.2f}%", True, COLORS['text_dim'])
            screen.blit(desv_text, (result_box.centerx - desv_text.get_width()//2, result_box.y + 100))
            
            if media > 0.2:
                status = "CUIDADO: ALTA PROBABILIDADE DE ESTOURO!"
                cor_status = COLORS['alert']
            else:
                status = "OK: REDE ROBUSTA CONTRA ENTUPIMENTOS"
                cor_status = COLORS['success']
            
            status_text = font.render(status, True, cor_status)
            screen.blit(status_text, (result_box.centerx - status_text.get_width()//2, result_box.y + 140))
            
            instr = font_small.render("Clique ou pressione qualquer tecla para continuar", True, COLORS['text_dim'])
            screen.blit(instr, (WIDTH//2 - 200, HEIGHT - 50))
            
            pygame.display.flip()
            clock.tick(30)
    
    def executar(self):
        """Executa a simulação com visualização"""
        relogio = pygame.time.Clock()
        
        while self.simulando:
            mouse_pos = pygame.mouse.get_pos()
            mouse_clicado = False
            
            # Processa eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_SPACE:
                        self.pausado = not self.pausado
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_clicado = True
            
            if not self.pausado:
                # Desenha estado atual e obtém botão de pausa
                btn_pause = self.desenhar(screen)
                
                # Verifica clique no botão de pausa
                if mouse_clicado and btn_pause.collidepoint(mouse_pos):
                    self.pausado = not self.pausado
                
                pygame.display.flip()
                
                # Avança para próxima simulação
                if not self.proxima_simulacao():
                    break
                
                # Controla velocidade
                relogio.tick(25)
            else:
                # Se pausado, apenas desenha e espera
                btn_pause = self.desenhar(screen)
                
                if mouse_clicado and btn_pause.collidepoint(mouse_pos):
                    self.pausado = not self.pausado
                
                pygame.display.flip()
                relogio.tick(10)
        
        # Mostra resultados finais
        self.mostrar_resultados()
        return True

# ============================
# MAIN
# ============================

def main():
    while True:
        # Menu inicial
        opcao, configs = tela_menu()
        
        if opcao is None:
            break
        
        qtd_ensaios, alfa, chance, p_limite = configs
        
        # Seleciona rede
        if opcao == "simples":
            rede = rede_simples
            nome = "REDE SIMPLES (4 nós, 4 arestas)"
        else:
            rede = rede_media
            nome = "REDE MÉDIA (8 nós, 12 arestas)"
        
        # Executa simulação
        sim = SimulacaoVisual(
            rede_func=rede,
            nome_rede=nome,
            qtd_ensaios=int(qtd_ensaios),
            p_max_limite=p_limite,
            alfa=alfa,
            chance_entupimento=chance
        )
        
        if not sim.executar():
            break
    
    pygame.quit()

if __name__ == "__main__":
    main()
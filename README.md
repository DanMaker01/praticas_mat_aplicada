# Problemas em Redes

$$
Zx=b
$$


## Instruções

Para executar os códigos é necessário instalar as seguintes bibliotecas:

```
pip install numpy scipy
```

# P1 - Vazões e Pressões na Rede

Resolve (encontra pessões e vazões) de uma Rede qualquer

# P2 - Entupimento

Entupimentos aleatórios (chance r da resistência de uma aresta qqr ser multiplicada por alfa) podem fazer a pressão explodir (passar do limite máximo).

Dada a rede, qual a chance de explodir? (Monte Carlo)

# P3 - Manutenção

r = [0.2, 0.1, 0.06, 0.04, 0.03, 0.02, 0.01]

queremos:

(i) rodar para cada r,

(ii) verificar a chance de faltar pressão (p_i < p_min) em cada ensaio e

(iii) decidir quantas manutenções/ano para que falte pressão menos que x% das vezes, em nenhum ensaio

# P4 - Otimização

suponha:

```
QTD_CANOS_BAIXA_RESIST = 2
BAIXA_RESIST = 0.1
```

queremos:

min SOMA ( | pi - pj | )

sujeito a poder a posição dos canos e

algo mais? (inteira? combinatória?)

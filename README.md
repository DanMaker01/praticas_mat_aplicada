# Problemas em Redes

## Exemplo (n=4, m=3)

$$
Zx=b
$$

$$
\left[
\begin{array}{cc}
C_{n\times m} & F_{n\times n}  \\
I_{m\times m} & D_{m\times n}  \\
\end{array}
\right]

\left[
\begin{array}{c}
\textbf{p} \\
\textbf{q} \\
\end{array}
\right]
=
\left[
\begin{array}{c}
\textbf{Q} \\
\textbf{0} \\
\end{array}
\right]
$$

$$
\left[
\begin{array}{ccc:cccc}
c_{11} & c_{12} & c_{13} & f_{14}
& f_{15} & f_{16} & f_{17} \\
c_{21} & c_{22} & c_{23} & f_{24}
& f_{25} & f_{26} & f_{27} \\
c_{31} & c_{32} & c_{33} & f_{34}
& f_{35} & f_{36} & f_{37} \\
c_{41} & c_{42} & c_{43} & f_{44}
& f_{45} & f_{46} & f_{47} \\
\hdashline
1 & 0 & 0 & d_{54}
& d_{55} & d_{56} & d_{57} \\
0 & 1 & 0 & d_{64}
& d_{65} & d_{66} & d_{67} \\
0 & 0 & 1 & d_{74}
& d_{75} & d_{76} & d_{77}
\end{array}
\right]
\left[
\begin{array}{c}
p_{1} \\
p_{2} \\
p_{3} \\
p_{4} \\
\hdashline
q_{5} \\
q_{6} \\
q_{7}
\end{array}
\right]
=
\left[
\begin{array}{c}
Q_{1} \\
Q_{2} \\
Q_{3} \\
Q_{4} \\
\hdashline
0 \\
0 \\
0 \\
\end{array}
\right]
$$

$$
c:
\newline

f:
\newline

d:
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

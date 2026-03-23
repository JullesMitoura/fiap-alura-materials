Este documento apresenta o passo a passo completo para obter os parâmetros β que minimizam o SSE (Soma dos Quadrados dos Erros) na regressão linear.


## 1. Contexto

- **Modelo:** ŷ = Xβ (forma matricial)
- **Objetivo:** Escolher β que minimize o SSE
- **SSE:** SSE = Σᵢ(yᵢ − ŷᵢ)²
- **Equivalência:** Minimizar SSE ⟺ Maximizar R²


## 2. SSE em notação matricial

- Vetor de erros: **e = y − Xβ**
- SSE = **eᵀe** = **(y − Xβ)ᵀ(y − Xβ)**

Expandindo o produto:

$$
\text{SSE} = (\mathbf{y}^T - \boldsymbol{\beta}^T\mathbf{X}^T)(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})
$$

$$
\text{SSE} = \mathbf{y}^T\mathbf{y} - \mathbf{y}^T\mathbf{X}\boldsymbol{\beta} - \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}
$$

Como **yᵀXβ** é escalar, então **yᵀXβ = (yᵀXβ)ᵀ = βᵀXᵀy**. Logo:

$$
\boxed{\text{SSE} = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}}
$$


## 3. Derivada em relação a β

Para encontrar o mínimo: $\displaystyle\frac{\partial \text{SSE}}{\partial \boldsymbol{\beta}} = \mathbf{0}$

**Regras de derivação matricial utilizadas:**

| Expressão | Derivada em relação a β |
|-----------|-------------------------|
| yᵀy (constante) | 0 |
| βᵀa (a é vetor) | a |
| βᵀAβ (A simétrica) | 2Aβ |

**Aplicando em cada termo:**

1. $\displaystyle\frac{\partial}{\partial \boldsymbol{\beta}}(\mathbf{y}^T\mathbf{y}) = \mathbf{0}$

2. $\displaystyle\frac{\partial}{\partial \boldsymbol{\beta}}(-2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y}) = -2\mathbf{X}^T\mathbf{y}$

3. $\displaystyle\frac{\partial}{\partial \boldsymbol{\beta}}(\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}) = 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}$

**Resultado:**

$$
\frac{\partial \text{SSE}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}
$$


## 4. Equação normal e solução

Igualando a zero:

$$
2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} - 2\mathbf{X}^T\mathbf{y} = \mathbf{0}
$$

$$
\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}
$$

Assumindo **(XᵀX)⁻¹** existe (colunas de X linearmente independentes):

$$
\boxed{\boldsymbol{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}
$$


## 5. Estrutura da matriz X

A matriz **X** (design matrix) deve incluir:

- **Coluna de 1s** para o termo constante (intercepto β₀)
- **Colunas das variáveis** independentes

Exemplo para uma variável:

$$
\mathbf{X} = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}
\quad
\boldsymbol{\beta} = \begin{bmatrix} \beta_0 \\ \beta_1 \end{bmatrix}
$$


## 6. Verificação (é mínimo?)

Hessiana:

$$
\frac{\partial^2 \text{SSE}}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T} = 2\mathbf{X}^T\mathbf{X}
$$

**XᵀX** é semi-definida positiva (definida positiva se X tem posto completo). Portanto, o ponto crítico é um **mínimo global**.


## 7. Implementação em Python (resumo)

```python
import numpy as np

# Dados: X (com coluna de 1s) e y
X = np.column_stack([np.ones(len(x)), x])  # ou suas variáveis

# Equação normal
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# Ou usando lstsq (mais estável numericamente)
beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
```

---

## Referências rápidas

- **Normal equation:** β = (XᵀX)⁻¹Xᵀy
- **Condição:** X com colunas L.I. (posto completo)
- **Alternativa:** Gradiente descendente quando n ou p são grandes
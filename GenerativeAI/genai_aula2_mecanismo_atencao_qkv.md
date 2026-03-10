Este documento explica, de forma didatica, o mecanismo de atencao apresentado em `genai_aula2_attention_scratch.ipynb`, com foco especial no papel de `Q`, `K` e `V`.


## 1. Contexto

- **Problema central:** em sequencias de texto, cada token depende do contexto dos outros tokens.
- **Limitacao de abordagens simples:** processar token por token pode perder dependencias longas.
- **Ideia da atencao:** para cada token, calcular **quais outros tokens sao mais relevantes** naquele momento.
- **Resultado pratico:** o modelo gera representacoes contextualizadas, e nao apenas embeddings isolados.

Em uma frase, o mecanismo de atencao responde: "para entender este token atual, em quais tokens da sequencia devo prestar mais atencao?".


## 2. Matriz de entrada e projecoes lineares

Antes da atencao, temos a matriz de entrada:

$$
X \in \mathbb{R}^{n \times d_{\text{model}}}
$$

onde:

- $n$ = numero de tokens da sequencia;
- $d_{\text{model}}$ = dimensao do embedding (ou representacao interna).

A partir de $X$, aprendemos tres projecoes lineares:

$$
\begin{aligned}
Q &= XW_Q \\
K &= XW_K \\
V &= XW_V
\end{aligned}
$$

com:

$$
W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}
$$

Assim, para cada token, obtemos tres vetores diferentes:

- **Query (Q):** "o que este token esta procurando?";
- **Key (K):** "que informacao este token oferece para ser encontrada?";
- **Value (V):** "qual conteudo efetivamente sera combinado na saida?".


## 3. Intuicao de Q, K e V

Uma intuicao util:

- `Q` funciona como uma **pergunta** feita por um token;
- `K` funciona como um **indice/rotulo** de cada token candidato;
- `V` funciona como o **conteudo** associado a esse candidato.

O modelo compara a pergunta (`Q`) com os indices (`K`) para decidir pesos de importancia. Depois, usa esses pesos para combinar os conteudos (`V`).

Se dois tokens sao fortemente relacionados no contexto, o produto entre seus vetores de `Q` e `K` tende a ser maior, aumentando o peso de atencao entre eles.


## 4. Calculo da atencao (Scaled Dot-Product Attention)

O mecanismo central e:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Passo a passo:

1. **Escores brutos (similaridade):**

$$
S = QK^T
$$

2. **Escala para estabilidade numerica e gradientes:**

$$
\tilde{S} = \frac{S}{\sqrt{d_k}}
$$

3. **Normalizacao com softmax (linha a linha):**

$$
A = \mathrm{softmax}(\tilde{S})
$$

4. **Combinacao ponderada dos valores:**

$$
H = AV
$$

onde:

- $A$ e a matriz de pesos de atencao (cada linha soma 1);
- $H$ e a saida contextualizada da camada de atencao.


## 5. Por que dividir por $\sqrt{d_k}$?

Sem a divisao por $\sqrt{d_k}$, os valores de $QK^T$ podem crescer muito quando $d_k$ aumenta. Isso torna o `softmax` muito "pontudo" (quase one-hot), o que pode:

- saturar gradientes;
- dificultar treinamento;
- reduzir estabilidade numerica.

A escala:

$$
\frac{QK^T}{\sqrt{d_k}}
$$

mantem os logits em uma faixa mais controlada e melhora o comportamento do treinamento.


## 6. Origem de Q, K e V no Transformer

Dependendo do bloco, a origem de `Q`, `K` e `V` muda:

- **Self-attention no encoder:** `Q`, `K` e `V` vem da mesma entrada $X$.
- **Self-attention no decoder:** `Q`, `K` e `V` tambem vem da entrada do decoder (com mascara causal).
- **Cross-attention no decoder:** `Q` vem do decoder, enquanto `K` e `V` vem da saida do encoder.

Isso permite que o modelo:

- entenda relacoes internas da propria sequencia (self-attention);
- conecte a sequencia de saida com a de entrada (cross-attention).


## 7. Softmax no contexto da atencao

A funcao softmax para cada elemento $i$ e:

$$
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

No mecanismo de atencao, ela transforma escores em pesos probabilisticos, isto e, valores entre 0 e 1 que somam 1 por consulta.

Interpretacao:

- peso alto $\Rightarrow$ token mais relevante para a consulta atual;
- peso baixo $\Rightarrow$ token menos relevante.


## 8. Exemplo conceitual rapido

Suponha uma sequencia com 3 tokens. Para o token 1 (consulta), os pesos de atencao poderiam ser:

$$
[0.10,\ 0.75,\ 0.15]
$$

Isso significa:

- 10% de importancia para o token 1;
- 75% para o token 2;
- 15% para o token 3.

A saida para o token 1 sera uma combinacao dos vetores de `V` nesses pesos:

$$
h_1 = 0.10\,v_1 + 0.75\,v_2 + 0.15\,v_3
$$

Logo, a representacao final de um token passa a carregar informacao de outros tokens relevantes do contexto.


## 9. Relacao com a implementacao da aula

No notebook, a atencao e implementada com os mesmos blocos conceituais:

- projecoes para gerar `Q`, `K`, `V`;
- calculo de escores com produto escalar;
- escalonamento por $\sqrt{d_k}$;
- `softmax` para pesos normalizados;
- multiplicacao dos pesos por `V`.

Mesmo em versao simplificada (single-head), esse fluxo ja reproduz o coracao do Transformer.


---

## Referencias rapidas

- **Entrada:** $X \in \mathbb{R}^{n \times d_{\text{model}}}$
- **Projecoes:** $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- **Atencao:** $\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- **Papel de Q/K/V:** consulta, indice de correspondencia e conteudo combinado
- **Efeito principal:** representacoes contextualizadas token-a-token

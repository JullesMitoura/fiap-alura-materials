# Bibliotecas — Aula 2: Mecanismo de Atenção do Zero

Mapeamento das bibliotecas em `requirements.txt` e sua utilidade no notebook da aula 2.

---

## Notebooks cobertos

| Notebook | Tema |
|---|---|
| `genai_aula2_attention_scratch.ipynb` | Implementação do mecanismo de atenção do zero com NumPy + classificador treinado com embeddings da OpenAI |

---

## Contexto geral

A aula 2 é uma continuação direta da aula 1: enquanto na aula 1 o Transformer foi construído com PyTorch (que encapsula as operações internas em módulos prontos como `nn.MultiheadAttention`), aqui o objetivo é abrir a "caixa preta" e implementar **cada operação matemática manualmente**, apenas com NumPy. A segunda parte do notebook substitui os embeddings aleatórios por vetores semânticos reais obtidos via API da OpenAI.

---

## `numpy`

Usado em: `genai_aula2_attention_scratch.ipynb` — backbone de toda a implementação

NumPy é a única dependência para a construção do Transformer do zero. Todas as operações que o PyTorch realiza internamente são reproduzidas explicitamente:

### Camada de Embedding

```python
embed = np.random.randn(vocab_size, dim_model)
return np.array([embed[i] for i in input_ids])
```

A matriz de embedding é inicializada com valores aleatórios normalmente distribuídos (`np.random.randn`). Cada token é mapeado para uma linha dessa matriz — equivalente ao que `nn.Embedding` faz no PyTorch.

### Função Softmax

```python
e = np.exp(x - np.max(x))
return e / e.sum(axis=-1).reshape(-1, 1)
```

Implementada manualmente com estabilidade numérica (subtrai o valor máximo antes da exponencial para evitar overflow). No contexto de atenção, converte os escores brutos (logits) em pesos normalizados que somam 1 — definindo o quanto cada token influencia cada posição.

### Scaled Dot-Product Attention

Implementa diretamente a fórmula do paper "Attention is All You Need":

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| Operação NumPy | Papel |
|---|---|
| `np.dot(Q, K.T)` | Produto escalar entre queries e chaves — calcula a afinidade entre cada par de posições |
| `K.shape[-1]` | Obtém `d_k` para a escala — evita que os escores cresçam demais com dimensões altas |
| `/ np.sqrt(depth)` | Divisão pela raiz de `d_k` — normaliza os escores antes do softmax |
| `softmax(logits)` | Converte escores em pesos de atenção |
| `np.dot(attention_weights, V)` | Pondera os vetores de valor pelos pesos — saída final da atenção |

### Camada Linear + Softmax

```python
weights = np.random.randn(dim_model, vocab_size)
logits = np.dot(input, weights)
return softmax(logits)
```

Projeta a saída da atenção (dimensão `dim_model`) para o espaço do vocabulário (dimensão `vocab_size`) e aplica softmax — equivalente a `nn.Linear` + `nn.Softmax` do PyTorch.

### Modelo completo

O pipeline do `model()` encadeia as quatro etapas acima e finaliza com `np.argmax` para selecionar o token de maior probabilidade em cada posição:

```
input_ids → embeddings → scaled_dot_product_attention → linear_softmax → np.argmax → token predito
```

### Pré-processamento e treinamento com embeddings OpenAI

Na segunda parte, NumPy continua como base para:

- `np.array([item.embedding for item in response.data], dtype=np.float32)` — converte a resposta da API em matrizes numéricas
- `np.zeros((max_len, emb_dim))` — cria sequências com padding para alinhar comprimentos
- `np.stack(X_seq, axis=0)` — empilha as amostras em um tensor 3D `(N, T, D)` para o classificador
- `np.random.randint` — gera sequências de entrada para testes do modelo puro

---

## `openai`

Usado em: `genai_aula2_attention_scratch.ipynb` — segunda parte (classificador com embeddings reais)

Após construir o Transformer do zero com valores aleatórios, o notebook demonstra que a qualidade das representações vetoriais é crítica para o desempenho. Para isso, substitui os embeddings aleatórios por vetores semânticos reais da API da OpenAI.

**Modelo usado:** `text-embedding-3-small`

**Função implementada:**

```python
def embeddings_openai(input_text, model="text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    return vectors[0] if is_single_input else vectors
```

- Aceita tanto uma string única quanto uma lista de strings (batch).
- Retorna um vetor 1D (`shape: (1536,)`) para entrada única ou uma matriz 2D para batch.
- É usada com cache (`embedding_cache`) para evitar chamadas repetidas para os mesmos tokens durante o pré-processamento do dataset.

**Dataset:** `sentimentos_data.csv` com 180 amostras em 3 classes (`felicidade`, `nulo`, `tristeza`). Cada texto é tokenizado com `re.findall`, seus tokens são convertidos em embeddings via API, e as sequências são truncadas em `max_len=8` tokens com padding por zeros.

---

## `python-dotenv`

Usado em: `genai_aula2_attention_scratch.ipynb`

Carrega a variável `OPENAI_API_KEY` do arquivo `.env` antes de instanciar o cliente `OpenAI`. Sem isso, a chave de API precisaria ser inserida diretamente no código — o que exporia credenciais em repositórios.

```python
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
```

O parâmetro `override=True` garante que os valores do `.env` sobrescrevam variáveis de ambiente já definidas no sistema — útil para trocar de chave sem reiniciar o ambiente.

---

## Comparação com a Aula 1

| Componente | Aula 1 (PyTorch) | Aula 2 (do zero) |
|---|---|---|
| Embedding | `nn.Embedding` | `np.random.randn` + indexação |
| Atenção | `nn.MultiheadAttention` | `np.dot(Q, K.T) / sqrt(d_k)` + softmax manual |
| Feed-forward / projeção | `nn.Linear` + `nn.ReLU` | `np.dot(input, weights)` + softmax manual |
| Otimização | `AdamW` + backpropagation automático | Sem treino (pesos aleatórios) / embeddings OpenAI |
| Embeddings | Aleatórios, aprendidos no treino | Aleatórios (parte 1) → semânticos via API (parte 2) |

---

## Resumo por notebook

| Biblioteca      | `aula2_attention_scratch` |
|-----------------|:-------------------------:|
| `numpy`         | ✓                         |
| `openai`        | ✓                         |
| `python-dotenv` | ✓                         |

# Bibliotecas — Aula 1: Mecanismo de Atenção com PyTorch

Mapeamento das bibliotecas em `requirements.txt` e sua utilidade nos notebooks da aula 1.

---

## Notebooks cobertos

| Notebook | Tema |
|---|---|
| `genai_aula1_attention_torch.ipynb` | Construção, treino e avaliação do Transformer com PyTorch |
| `genai_aula1_model_test.ipynb` | Inferência e interpretação do modelo treinado |

---

## `torch` — PyTorch

Usado em: ambos os notebooks

Biblioteca central de toda a aula 1. Fornece a infraestrutura para definir a arquitetura do Transformer, realizar o treino e executar inferências. É a única dependência externa de ambos os notebooks.

### No notebook de treino (`genai_aula1_attention_torch.ipynb`)

**Definição da arquitetura com `torch.nn`:**

| Componente | Papel no modelo |
|---|---|
| `nn.Module` | Classe base da qual `Transformer` herda — define o contrato `forward()` |
| `nn.Embedding` | Converte índices de tokens em vetores densos (embeddings) de dimensão `embedding_dim` |
| `nn.MultiheadAttention` | Implementa o mecanismo de atenção multi-cabeça; recebe Q, K, V (todos iguais a `x` em self-attention) e retorna a saída combinada com pesos de atenção |
| `nn.ModuleList` | Registra os `n_layers` blocos de atenção como parâmetros treináveis do modelo |
| `nn.Sequential` | Encadeia as camadas da rede feed-forward (Linear → ReLU → Linear) |
| `nn.Linear` | Camada totalmente conectada; usada na feed-forward e na projeção final sobre o vocabulário |
| `nn.ReLU` | Função de ativação entre as duas camadas lineares da feed-forward |
| `nn.CrossEntropyLoss` | Função de perda — calcula o erro entre os logits do modelo e o próximo token esperado |

**Conexão residual:** após cada bloco de atenção, a saída é somada à entrada (`x = x + attn_out`), padrão da arquitetura Transformer original.

**Treinamento:**

| Componente | Papel |
|---|---|
| `torch.optim.AdamW` | Otimizador com decaimento de pesos; atualiza os parâmetros a cada batch |
| `torch.utils.data.Dataset` | Interface base da classe `ToyDataset`, que gera pares (sequência de entrada, próximo token alvo) |
| `torch.utils.data.DataLoader` | Agrupa as amostras em batches de tamanho 32, embaralha a cada época e itera o loop de treino |
| `torch.tensor` | Converte listas de IDs de token em tensores `long` para alimentar o modelo |

**Dataset sintético:** `torch.randint` gera 5.000 IDs aleatórios no intervalo `[0, vocab_size)`, que são segmentados em janelas de comprimento `seq_len=16` para criar os pares X/y de treino.

**Avaliação:**

| Métrica | Como é calculada |
|---|---|
| Loss média por token | `CrossEntropyLoss` acumulada por número de tokens |
| Perplexidade | `math.exp(loss)` — quanto mais próxima de 1, mais confiante o modelo |
| Acurácia top-1 | `logits.argmax(dim=-1)` comparado ao token real |
| Acurácia top-5 | `logits.topk(k=5)` — verifica se o token real está entre os 5 mais prováveis |

**Salvamento do checkpoint:** `torch.save` persiste `model.state_dict()` junto com os hiperparâmetros em `models/transformer_attention.pt`.

---

### No notebook de inferência (`genai_aula1_model_test.ipynb`)

A mesma classe `Transformer` é redefinida manualmente (sem importar o notebook anterior), e o checkpoint é restaurado com `torch.load`.

| Componente | Papel |
|---|---|
| `torch.load` | Carrega o checkpoint `.pt` e mapeia os tensores para o device disponível (`cpu` ou `cuda`) |
| `model.load_state_dict` | Restaura os pesos treinados na arquitetura reconstruída |
| `torch.no_grad` | Desliga o cálculo de gradientes durante a inferência, reduzindo uso de memória |
| `torch.softmax` | Converte os logits da última posição da sequência em distribuição de probabilidades sobre o vocabulário |
| `torch.topk` | Retorna os `k` tokens de maior probabilidade — base do ranking top-k exibido ao usuário |
| `torch.manual_seed` | Garante reprodutibilidade na configuração do device |

**Fluxo de inferência:**
1. Uma sequência de 16 tokens é passada ao modelo.
2. O logit da última posição é extraído (`logits[:, -1, :]`).
3. Softmax gera probabilidades sobre todo o vocabulário.
4. `topk` seleciona os 5 candidatos mais prováveis.
5. O resultado exibe se o token esperado foi acertado no top-1 e se aparece no top-5.

---

## `matplotlib`

Usado em: `genai_aula1_attention_torch.ipynb`

Utilizada exclusivamente para plotar a curva de evolução da loss ao longo das 1.000 épocas de treino. O gráfico (`plt.plot(loss_history)`) permite observar a convergência do modelo e identificar se houve estabilização ou overfitting.

---

## Resumo por notebook

| Biblioteca    | `aula1_attention_torch` | `aula1_model_test` |
|---------------|:-----------------------:|:------------------:|
| `torch`       | ✓                       | ✓                  |
| `matplotlib`  | ✓                       |                    |

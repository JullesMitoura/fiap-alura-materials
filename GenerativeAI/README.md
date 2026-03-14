# GenAI & Advanced Nets

**PhD. Julles Mitoura**

Repositório da disciplina de Inteligência Artificial Generativa e Redes Avançadas. Os notebooks percorrem uma trilha progressiva: da implementação manual de mecanismos internos de Transformers até o consumo de LLMs em produção via API.

---

## Estrutura da disciplina

```
GenAI/
├── genai_aula1_attention_torch.ipynb               # Aula 1a — Transformer com PyTorch
├── genai_aula1_model_test.ipynb                    # Aula 1b — Inferência e teste do modelo
├── genai_aula2_attention_scratch.ipynb             # Aula 2  — Atenção do zero (NumPy + OpenAI)
├── genai_aula3_1_models_lstm.ipynb                 # Aula 3a — Classificação com LSTM
├── genai_aula3_2_models_fcnn.ipynb                 # Aula 3b — Classificação com FCNN + TF-IDF
├── genai_aula3_3_models_transformer.ipynb          # Aula 3c — Fine-tuning de BERT
├── genai_aula3_4_models_evaluate.ipynb             # Aula 3d — Avaliação comparativa
├── genai_aula4_llm_api.ipynb                       # Aula 4  — Consumo de LLMs via API
├── requirements.txt
├── .env.example                                    # OPENAI_API_KEY (não versionar)
└── suporte/                                        # Documentação de bibliotecas por aula
```

---

## Aula 1 — O Mecanismo de Atenção com PyTorch

### `genai_aula1_attention_torch.ipynb` — Construção e treino

Apresenta a arquitetura Transformer proposta no paper *"Attention is All You Need"* (2017) e implementa um modelo completo com PyTorch, do zero.

**O que é abordado:**
- Visão geral da arquitetura Transformer: embeddings, atenção multi-cabeça, feed-forward, conexões residuais e projeção sobre o vocabulário
- Implementação da classe `Transformer` com `torch.nn`: `Embedding`, `MultiheadAttention`, `ModuleList`, `Linear`, `ReLU`, `Sequential`
- Criação de um dataset sintético com `torch.utils.data.Dataset` e `DataLoader` para treino de predição do próximo token
- Loop de treino com `CrossEntropyLoss` e otimizador `AdamW` ao longo de 1.000 épocas
- Avaliação com loss média, perplexidade, acurácia top-1 e acurácia top-5
- Salvamento do checkpoint treinado em `models/transformer_attention.pt`

**Bibliotecas:** `torch`, `matplotlib`

---

### `genai_aula1_model_test.ipynb` — Inferência e teste

Consome o checkpoint gerado no notebook anterior e demonstra o fluxo completo de inferência.

**O que é abordado:**
- Reconstrução da arquitetura e restauração dos pesos com `torch.load` e `load_state_dict`
- Inferência sem gradientes com `torch.no_grad`
- Geração de ranking top-k: `softmax` sobre os logits da última posição e `topk` para os tokens mais prováveis
- Interpretação dos resultados: acerto top-1 e presença do token esperado no top-5

**Bibliotecas:** `torch`

---

## Aula 2 — O Mecanismo de Atenção do Zero

### `genai_aula2_attention_scratch.ipynb`

Reimplementa os componentes internos do Transformer manualmente, sem usar nenhum framework de deep learning — apenas NumPy. Na segunda parte, substitui os embeddings aleatórios por vetores semânticos reais da API da OpenAI.

**O que é abordado:**

**Parte 1 — Implementação manual com NumPy:**
- Camada de embedding como indexação em uma matriz aleatória
- Função `softmax` com estabilidade numérica (subtração do máximo antes da exponencial)
- `scaled_dot_product_attention`: produto escalar QKᵀ, escala por √dₖ, softmax e ponderação de V — fórmula completa do paper
- Camada `linear_softmax`: projeção para o espaço do vocabulário seguida de softmax
- Modelo completo em 5 etapas: embedding → self-attention → projeção linear → argmax → token predito

**Parte 2 — Classificador com embeddings reais:**
- Geração de embeddings semânticos via `text-embedding-3-small` (OpenAI API)
- Cache de embeddings por token para evitar chamadas redundantes
- Pré-processamento do dataset `sentimentos_data.csv` (180 amostras, 3 classes): tokenização com `re`, padding com NumPy, empilhamento em tensor `(N, T, D)`
- Encoder Transformer implementado com NumPy para classificação de sentimentos

**Bibliotecas:** `numpy`, `openai`, `python-dotenv`

> Material de suporte: `suporte/genai_aula2_mecanismo_atencao_qkv.md` — detalhamento matemático de Q, K e V

---

## Aula 3 — Modelos de NLP para Classificação de Texto

Quatro notebooks que treinam e comparam três arquiteturas diferentes para a mesma tarefa de classificação de texto, partindo de abordagens clássicas até o fine-tuning de modelos pré-treinados.

---

### `genai_aula3_1_models_lstm.ipynb` — LSTM

Rede neural recorrente bidirecional para classificação de texto.

**O que é abordado:**
- Pré-processamento com spaCy: tokenização, lematização e remoção de stopwords
- Codificação de rótulos (`LabelEncoder`) e balanceamento com `compute_class_weight`
- Tokenização e padding de sequências com `Tokenizer` e `pad_sequences` do Keras
- Arquitetura: `Embedding` → `Bidirectional(LSTM)` → `Dropout` → `Dense`
- Treino com `EarlyStopping` e `CrossEntropyLoss` ponderada
- Avaliação: `classification_report`, `confusion_matrix`, `f1_score`, curvas de treino
- Salvamento do modelo e tokenizador com `pickle`

**Bibliotecas:** `tensorflow`, `scikit-learn`, `spacy`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_2_models_fcnn.ipynb` — FCNN + TF-IDF

Rede feed-forward sobre vetores TF-IDF — abordagem mais simples, sem recorrência.

**O que é abordado:**
- Mesmo pipeline de pré-processamento com spaCy do notebook anterior
- Vetorização com `TfidfVectorizer`: cada texto vira um vetor esparso onde cada dimensão é um termo ponderado pela sua relevância no corpus
- Arquitetura: `Dense` → `Dropout` → `Dense` (camadas totalmente conectadas, sem embedding)
- Comparação implícita com LSTM: menor complexidade arquitetural, entrada estruturada
- Salvamento do vetorizador TF-IDF e do modelo

**Bibliotecas:** `tensorflow`, `scikit-learn`, `spacy`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_3_models_transformer.ipynb` — Fine-tuning de BERT

Fine-tuning de um modelo BERT pré-treinado com a biblioteca HuggingFace Transformers.

**O que é abordado:**
- Diferença entre treinar do zero (LSTM, FCNN) e fine-tuning de modelo pré-treinado (BERT)
- Tokenização com `AutoTokenizer` no formato WordPiece esperado pelo BERT
- Conversão do DataFrame em `Dataset` HuggingFace para uso com o `Trainer`
- `DataCollatorWithPadding` para padding dinâmico dos batches
- Configuração do fine-tuning com `TrainingArguments`: épocas, learning rate, batch size, avaliação periódica
- Treino com `Trainer` e avaliação com métricas do scikit-learn
- Salvamento do modelo e tokenizador fine-tunados

**Bibliotecas:** `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_4_models_evaluate.ipynb` — Avaliação Comparativa

Carrega os três modelos treinados (LSTM, FCNN e BERT) e os avalia sobre o mesmo conjunto de teste para uma comparação justa.

**O que é abordado:**
- Reaplicação do pipeline de pré-processamento com spaCy para LSTM e FCNN
- Carregamento do LSTM com `load_model` e reconstrução das sequências com `pad_sequences`
- Carregamento do FCNN com `load_model` e vetorização com o `TfidfVectorizer` salvo
- Carregamento do BERT com `pipeline` do HuggingFace para inferência direta
- Métricas unificadas: accuracy, F1 macro, relatório de classificação e matriz de confusão para cada modelo
- Gráfico comparativo de performance entre as três arquiteturas

**Bibliotecas:** `tensorflow`, `transformers`, `torch`, `spacy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

## Aula 4 — Consumo de LLMs via API

### `genai_aula4_llm_api.ipynb`

Muda o foco de construção/treino de modelos para **consumo de LLMs em produção**. Aborda o Chat Completions da OpenAI com detalhamento de cada parâmetro de geração e boas práticas de engenharia.

**O que é abordado:**

**Fundamentos:**
- Estrutura das mensagens: roles `system`, `user` e `assistant`
- Anatomia da resposta: `choices`, `finish_reason`, `usage` (tokens de entrada, saída e total)

**Parâmetros de geração** (cada um com explicação teórica + experimento comparativo):

| Parâmetro | O que controla |
|---|---|
| `model` | Escolha do modelo: trade-off entre capacidade e custo |
| `temperature` | Aleatoriedade na seleção de tokens (0 = determinístico, 2 = máxima criatividade) |
| `max_tokens` | Limite de tokens na saída; impacto no `finish_reason` |
| `top_p` | Nucleus sampling: filtra os tokens de menor probabilidade após o softmax |
| `frequency_penalty` | Penaliza tokens proporcionalmente à frequência de aparição |
| `presence_penalty` | Penaliza tokens pela simples presença (encoraja novos temas) |
| `stop` | Strings de parada — interrompe a geração ao encontrá-las |
| `n` | Número de respostas independentes geradas por chamada |
| `seed` | Reprodutibilidade — associado ao `system_fingerprint` |
| `stream` | Respostas progressivas token a token para interfaces em tempo real |

**Boas práticas:**
- System prompt fraco vs. forte: comparação de impacto na qualidade
- Estimativa de custo por chamada a partir de `response.usage`
- Tratamento de erros com retry exponencial (`RateLimitError`, `AuthenticationError`, `BadRequestError`)
- Saídas estruturadas em JSON com `response_format={"type": "json_object"}`
- Gestão de histórico para conversas com múltiplos turnos (API stateless)

**Bibliotecas:** `openai`, `python-dotenv`

---

## Pré-requisitos

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
python -m spacy download pt_core_news_sm   # modelo spaCy em português
```

Crie um arquivo `.env` na raiz com sua chave de API:

```
OPENAI_API_KEY=sk-...
```

---

## Documentação de bibliotecas

Cada aula possui um arquivo de suporte detalhando a função de cada biblioteca no contexto dos notebooks:

| Arquivo | Aulas |
|---|---|
| `suporte/genai_aula1_libs.md` | Aula 1a e 1b |
| `suporte/genai_aula2_libs.md` | Aula 2 |
| `suporte/genai_aula3_libs.md` | Aulas 3a, 3b, 3c e 3d |
| `suporte/genai_aula4_libs.md` | Aula 4 |
| `suporte/genai_aula2_mecanismo_atencao_qkv.md` | Detalhamento matemático de Q, K e V |

---

## Trilha de aprendizado

```
Aula 1  →  Transformer com PyTorch (implementação + treino + inferência)
   ↓
Aula 2  →  Mesma arquitetura do zero, só com NumPy (abre a "caixa preta")
            + embeddings reais via API OpenAI
   ↓
Aula 3  →  Modelos de NLP aplicados (LSTM → FCNN → BERT fine-tuning → comparação)
   ↓
Aula 4  →  Consumo de LLMs em produção: parâmetros, boas práticas, API
```

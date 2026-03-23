<div align="center">

<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>


<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg"
     alt="Python"
     width="48"
     height="48"/>
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original.svg"
     alt="Jupyter"
     width="48"
     height="48"/>
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original.svg"
     alt="PyTorch"
     width="48"
     height="48"/>
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg"
     alt="TensorFlow"
     width="48"
     height="48"/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)


<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>

</div>

# IA Generativa e Redes Avançadas
> PhD. Julles Mitoura

Módulo prático de Inteligência Artificial Generativa. Os notebooks percorrem uma trilha progressiva: da implementação manual do mecanismo de atenção até o consumo de LLMs em produção — passando por fine-tuning eficiente com LoRA, classificação de texto e fundamentos de embeddings.

---

## Estrutura do Módulo

```
03_GenerativeAI/
├── genai_aula1_attention_torch.ipynb               # Aula 1a — Transformer com PyTorch
├── genai_aula1_model_test.ipynb                    # Aula 1b — Inferência e teste do modelo
├── genai_aula2_attention_scratch.ipynb             # Aula 2  — Atenção do zero (só NumPy)
├── genai_aula3_1_models_lstm.ipynb                 # Aula 3a — Classificação de sentimentos: BiLSTM
├── genai_aula3_2_models_fcnn.ipynb                 # Aula 3b — Classificação de sentimentos: FCNN + TF-IDF
├── genai_aula3_3_models_transformer.ipynb          # Aula 3c — Fine-tuning de DistilBERT
├── genai_aula3_4_models_evaluate.ipynb             # Aula 3d — Avaliação comparativa dos três modelos
├── genai_aula4_lora_vision.ipynb                   # Aula 4  — LoRA para Visão (ViT + Beans dataset)
├── genai_aula5_embedding.ipynb                     # Aula 5  — Modelos de Embedding
├── genai_aula6_llm_api_call.ipynb                  # Aula 6  — Consumo de LLMs via API
├── requirements.txt
├── .env.example                                    # OPENAI_API_KEY (não versionar)
└── suporte/                                        # Documentação de bibliotecas por aula
```

---

## Aula 1 — O Mecanismo de Atenção com PyTorch

### `genai_aula1_attention_torch.ipynb` — Construção e treino

Apresenta a arquitetura Transformer proposta no paper *"Attention is All You Need"* (2017) e implementa um modelo completo com PyTorch, do zero.

**O que é abordado:**
- Visão geral da arquitetura Transformer: embeddings, single-head e multi-head attention, feed-forward, conexões residuais e projeção sobre o vocabulário
- Implementação da classe `Transformer` com `torch.nn`: `Embedding`, `MultiheadAttention`, `LayerNorm`, `Linear`, `GELU`, `Sequential`
- Dataset sintético com `torch.utils.data.Dataset` e `DataLoader` para treino de predição do próximo token (vocab_size=1000, seq_len=16)
- Loop de treino com `CrossEntropyLoss` e otimizador `AdamW` ao longo de 1.000 épocas
- Avaliação: loss média, perplexidade, acurácia top-1 (99,31%) e top-5 (100%)
- Salvamento do checkpoint em `models/transformer_attention.pt`

**Bibliotecas:** `torch`, `matplotlib`

---

### `genai_aula1_model_test.ipynb` — Inferência e teste

Consome o checkpoint gerado no notebook anterior e demonstra o fluxo completo de inferência.

**O que é abordado:**
- Reconstrução da arquitetura e restauração dos pesos com `torch.load` e `load_state_dict`
- Inferência sem gradientes com `torch.no_grad`
- Ranking top-k: `softmax` sobre os logits da última posição e `topk` para os tokens mais prováveis
- Interpretação dos resultados: acerto top-1 e presença do token esperado no top-5

**Bibliotecas:** `torch`

---

## Aula 2 — O Mecanismo de Atenção do Zero

### `genai_aula2_attention_scratch.ipynb`

Reimplementa os componentes internos do Transformer manualmente — apenas NumPy, sem nenhum framework de deep learning.

**O que é abordado:**
- Camada de embedding como indexação em uma matriz aleatória
- `softmax` com estabilidade numérica (subtração do máximo)
- `scaled_dot_product_attention`: produto escalar QKᵀ, escala por √dₖ, softmax e ponderação de V
- Camada `linear_softmax`: projeção para o espaço do vocabulário
- Encoder e Decoder do zero; predição sobre sequências sintéticas (1–10 tokens)
- Modelo completo em 5 etapas: embedding → self-attention → projeção linear → argmax → token predito

**Bibliotecas:** `numpy`

> Material de suporte: `suporte/genai_aula2_mecanismo_atencao_qkv.md` — detalhamento matemático de Q, K e V

---

## Aula 3 — Modelos de NLP para Classificação de Texto

Quatro notebooks que treinam e comparam três arquiteturas para a mesma tarefa: classificação de sentimentos em 6 classes (anger, fear, joy, love, sadness, surprise) sobre 18.000 amostras.

---

### `genai_aula3_1_models_lstm.ipynb` — BiLSTM

Rede neural recorrente bidirecional para classificação de texto.

**O que é abordado:**
- Pré-processamento com spaCy: lematização, remoção de stopwords e pontuação
- Tokenização e padding de sequências com `Tokenizer` e `pad_sequences` do Keras (MAX_LEN=100)
- `LabelEncoder` e balanceamento com `compute_class_weight`
- Arquitetura: `Embedding(128)` → `Bidirectional(LSTM(64))` → `Dropout` → `Dense(64, ReLU)` → `Dense(6, Softmax)`
- Treino com `EarlyStopping` e cross-entropy ponderada
- Avaliação: `classification_report`, `confusion_matrix`, curvas de treino — F1-Macro ~86%

**Bibliotecas:** `tensorflow`, `scikit-learn`, `spacy`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_2_models_fcnn.ipynb` — FCNN + TF-IDF

Rede feed-forward sobre vetores TF-IDF — abordagem mais simples, sem recorrência.

**O que é abordado:**
- Mesmo pipeline de pré-processamento com spaCy
- `TfidfVectorizer` (unigramas + bigramas): 11.504 features esparsas
- Arquitetura: `Dense(256, SELU)` → `Dropout` → `Dense(128, SELU)` → `Dense(64, SELU)` → `Dense(6, Softmax)`
- Comparação implícita com BiLSTM: entrada estruturada, sem embedding
- F1-Macro ~86%, salvamento do vetorizador TF-IDF e do modelo

**Bibliotecas:** `tensorflow`, `scikit-learn`, `spacy`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_3_models_transformer.ipynb` — Fine-tuning de DistilBERT

Fine-tuning de DistilBERT pré-treinado com a biblioteca HuggingFace Transformers.

**O que é abordado:**
- DistilBERT: 40% menos parâmetros que BERT-base, 60% mais rápido, 97% da capacidade — via knowledge distillation
- Tokenização WordPiece com `AutoTokenizer`; sem pré-processamento manual de texto
- Conversão do DataFrame em `Dataset` HuggingFace e `DataCollatorWithPadding` para padding dinâmico
- Fine-tuning com `TrainingArguments` (lr=2e-5, weight_decay=0.01, 4 épocas) e `Trainer`
- F1-Macro ~87%, salvamento em `models/distilbert_sentiment/`

**Bibliotecas:** `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

### `genai_aula3_4_models_evaluate.ipynb` — Avaliação Comparativa

Carrega os três modelos treinados (BiLSTM, FCNN e DistilBERT) e avalia sobre o mesmo conjunto de teste.

**O que é abordado:**
- Reaplicação do pipeline spaCy para BiLSTM e FCNN; inferência direta com `pipeline` do HuggingFace para DistilBERT
- Métricas unificadas: accuracy, F1 macro, `classification_report` e `confusion_matrix` por modelo
- Análise de confiança por classe e comparação de padrões de erro
- Gráfico comparativo: FCNN 88,9%, BiLSTM 94,4%, DistilBERT 94,4%

**Bibliotecas:** `tensorflow`, `transformers`, `torch`, `spacy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

## Aula 4 — LoRA para Visão

### `genai_aula4_lora_vision.ipynb`

Demonstra fine-tuning eficiente de um Vision Transformer (ViT) usando LoRA — atualizando apenas ~0,2% dos parâmetros do modelo base.

**O que é abordado:**
- **Vision Transformer (ViT):** divisão de imagens em patches e processamento como sequência via atenção
- **LoRA (Low-Rank Adaptation):** injeção de matrizes de baixo rank nas projeções de atenção (query, value); congelamento do restante
- Configuração com `LoraConfig` da biblioteca `peft`: rank=8, alpha=16, target modules = query e value
- Dataset **Beans** (HuggingFace): 3 classes de doenças em plantas (angular_leaf_spot, bean_rust, healthy), ~1.000 imagens de treino
- Pré-processamento com `AutoImageProcessor` (redimensionamento para 224×224)
- Fine-tuning com `Trainer` (lr=2e-4, batch=16, 5 épocas); pode rodar em CPU (~15 min)
- Avaliação: `classification_report` e `confusion_matrix`

**Bibliotecas:** `transformers`, `peft`, `datasets`, `torch`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`

---

## Aula 5 — Modelos de Embedding

### `genai_aula5_embedding.ipynb`

Explora o que são tokens e embeddings — os blocos fundamentais de qualquer modelo de linguagem — e demonstra como utilizá-los na prática.

**O que é abordado:**
- **Tokens:** unidades atômicas de texto; tokenização BPE e WordPiece com prefixo `##` para subtokens
- **Embeddings:** representações vetoriais densas; por que evitar one-hot encoding em vocabulários grandes
- **Similaridade de cosseno:** medida de proximidade semântica no espaço vetorial
- **Redução de dimensionalidade:** PCA e t-SNE para visualização de clusters semânticos
- **Pesos congelados vs. treináveis:** `requires_grad=False` para transfer learning eficiente
- Construção de um modelo de embedding customizado com `torch.nn`
- Aplicações práticas: busca semântica, clustering, sistemas de recomendação

**Bibliotecas:** `transformers`, `torch`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`

---

## Aula 6 — Consumo de LLMs via API

### `genai_aula6_llm_api_call.ipynb`

Muda o foco de construção/treino de modelos para **consumo de LLMs em produção** via OpenAI API. Cada parâmetro de geração é explicado teoricamente e demonstrado com experimentos comparativos.

**O que é abordado:**

**Fundamentos:**
- Estrutura das mensagens: roles `system`, `user` e `assistant`
- Anatomia da resposta: `choices`, `finish_reason`, `usage` (tokens de entrada, saída e total)

**Parâmetros de geração:**

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

**Boas práticas:**
- System prompt fraco vs. forte: comparação de impacto na qualidade
- Estimativa de custo por chamada a partir de `response.usage`
- Tratamento de erros: `RateLimitError`, `AuthenticationError`, `BadRequestError`

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

Crie um arquivo `.env` na raiz com sua chave de API (necessário apenas para a Aula 6):

```
OPENAI_API_KEY=sk-...
```

| Biblioteca | Uso |
|---|---|
| `torch` | Framework principal de deep learning (Aulas 1, 2, 3c, 4, 5) |
| `tensorflow` | Treino de BiLSTM e FCNN (Aulas 3a, 3b, 3d) |
| `transformers` | DistilBERT, ViT, embeddings HuggingFace (Aulas 3c, 4, 5) |
| `peft` | LoRA e fine-tuning eficiente (Aula 4) |
| `datasets` | Carregamento de datasets HuggingFace (Aulas 3c, 4) |
| `spacy` | Pré-processamento NLP em português (Aulas 3a, 3b, 3d) |
| `scikit-learn` | Métricas, TF-IDF, normalização (Aulas 3, 4, 5) |
| `openai` | Chat Completions API (Aula 6) |
| `python-dotenv` | Gestão de variáveis de ambiente (Aula 6) |
| `numpy`, `matplotlib`, `seaborn`, `pandas` | Álgebra, visualização e manipulação de dados |

---

## Documentação de bibliotecas

Cada aula possui um arquivo de suporte detalhando a função de cada biblioteca no contexto dos notebooks:

| Arquivo | Aulas |
|---|---|
| `suporte/genai_aula1_libs.md` | Aula 1a e 1b |
| `suporte/genai_aula2_libs.md` | Aula 2 |
| `suporte/genai_aula2_mecanismo_atencao_qkv.md` | Detalhamento matemático de Q, K e V |
| `suporte/genai_aula3_libs.md` | Aulas 3a, 3b, 3c e 3d |
| `suporte/genai_aula4_libs.md` | Aula 4 |
| `suporte/genai_aula6_libs.md` | Aula 6 |

---

## Trilha de aprendizado

```
Aula 1  →  Transformer com PyTorch (implementação + treino + inferência)
   ↓
Aula 2  →  Mesma arquitetura do zero, só com NumPy (abre a "caixa preta")
   ↓
Aula 3  →  NLP aplicado: BiLSTM → FCNN+TF-IDF → DistilBERT fine-tuning → comparação
   ↓
Aula 4  →  LoRA: fine-tuning eficiente de Vision Transformer em imagens
   ↓
Aula 5  →  Embeddings: tokens, vetores semânticos, similaridade e visualização
   ↓
Aula 6  →  Consumo de LLMs em produção: parâmetros, boas práticas, OpenAI API
```

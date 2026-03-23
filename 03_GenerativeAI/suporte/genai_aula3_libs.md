# Bibliotecas — Aula 3: Modelos de NLP

Mapeamento das bibliotecas em `requirements.txt` e sua utilidade nos notebooks da aula 3.

---

## `torch` — PyTorch

Usado em: `genai_aula3_3_models_transformer.ipynb`, `genai_aula3_4_models_evaluate.ipynb`

Framework de deep learning que serve de base para os modelos do HuggingFace (Transformers). No notebook de Transformer, toda a fase de fine-tuning do modelo BERT é executada sobre tensores PyTorch. No notebook de avaliação, o modelo treinado é carregado e as inferências também são realizadas via PyTorch.

---

## `matplotlib`

Usado em: todos os notebooks

Biblioteca central de visualização. É utilizada para plotar as curvas de treinamento (loss/accuracy por época), os gráficos de distribuição de classes e, em conjunto com `seaborn`, as matrizes de confusão. No notebook de avaliação (`aula3_4`) também é usada para comparar as métricas dos três modelos lado a lado.

---

## `pandas`

Usado em: todos os notebooks

Responsável pela leitura e manipulação dos dados tabulares (CSV com textos e rótulos). Todas as etapas de exploração do dataset — verificação de classes, contagem de amostras, filtragem — são feitas com DataFrames pandas antes de passar os dados para os modelos.

---

## `scikit-learn`

Usado em: todos os notebooks

Oferece utilitários essenciais ao pipeline de ML:

- `train_test_split` — divisão treino/validação/teste em todos os modelos
- `LabelEncoder` — codificação dos rótulos de texto para inteiros (LSTM, FCNN, Transformer)
- `TfidfVectorizer` — vetorização TF-IDF dos textos no modelo FCNN (`aula3_2`)
- `compute_class_weight` — cálculo de pesos para lidar com desbalanceamento de classes (LSTM, FCNN)
- `classification_report`, `confusion_matrix`, `f1_score`, `accuracy_score` — métricas de avaliação em todos os notebooks

---

## `openai`

Usado em: não utilizado diretamente nos notebooks da aula 3

Biblioteca para integração com a API da OpenAI. Presente no `requirements.txt` para uso em outros notebooks da disciplina (ex.: `genai_aula1_model_test.ipynb`).

---

## `python-dotenv`

Usado em: não utilizado diretamente nos notebooks da aula 3

Carrega variáveis de ambiente de um arquivo `.env` (ex.: chave de API da OpenAI). Presente como dependência de suporte para os notebooks que consomem APIs externas.

---

## `seaborn`

Usado em: todos os notebooks

Complementa o `matplotlib` com visualizações estatísticas mais elaboradas. Seu principal uso na aula 3 é a geração de heatmaps para as matrizes de confusão, tornando mais fácil identificar padrões de erro de classificação de cada modelo.

---

## `spacy`

Usado em: `genai_aula3_1_models_lstm.ipynb`, `genai_aula3_2_models_fcnn.ipynb`, `genai_aula3_4_models_evaluate.ipynb`

Biblioteca de NLP usada na etapa de pré-processamento de texto. Realiza tokenização, remoção de stopwords e lematização dos textos antes de alimentar os modelos LSTM e FCNN. No notebook de avaliação, o mesmo pipeline de pré-processamento é aplicado aos textos de teste para garantir consistência com o que os modelos viram no treino.

---

## `tensorflow`

Usado em: `genai_aula3_1_models_lstm.ipynb`, `genai_aula3_2_models_fcnn.ipynb`, `genai_aula3_4_models_evaluate.ipynb`

Framework de deep learning utilizado para construir e treinar os modelos LSTM e FCNN via `tensorflow.keras`:

- **LSTM** (`aula3_1`): camadas `Embedding`, `LSTM`, `Bidirectional`, `Dense`, `Dropout`, `EarlyStopping`, `Tokenizer`, `pad_sequences`
- **FCNN** (`aula3_2`): camadas `Dense`, `Dropout`, `EarlyStopping` — modelo mais simples, sem recorrência, que recebe como entrada os vetores TF-IDF
- **Avaliação** (`aula3_4`): os modelos salvos são carregados com `load_model` e as sequências tokenizadas são reconstruídas com `pad_sequences`

---

## `tf_keras`

Usado em: `genai_aula3_1_models_lstm.ipynb`, `genai_aula3_2_models_fcnn.ipynb`, `genai_aula3_4_models_evaluate.ipynb` (dependência indireta)

Pacote standalone que expõe a API Keras de forma independente do TensorFlow. É requerido como dependência de compatibilidade pelo `tensorflow` em ambientes onde as versões do Keras e do TensorFlow precisam ser gerenciadas separadamente.

---

## `transformers`

Usado em: `genai_aula3_3_models_transformer.ipynb`, `genai_aula3_4_models_evaluate.ipynb`

Biblioteca da HuggingFace que disponibiliza modelos pré-treinados baseados na arquitetura Transformer. No notebook de fine-tuning (`aula3_3`), são usados:

- `AutoTokenizer` — tokenização dos textos no formato esperado pelo BERT
- `AutoModelForSequenceClassification` — modelo BERT pré-treinado adaptado para classificação de texto
- `TrainingArguments` e `Trainer` — API de alto nível para o loop de fine-tuning (épocas, learning rate, batch size, avaliação)
- `DataCollatorWithPadding` — padding dinâmico dos batches durante o treino

No notebook de avaliação (`aula3_4`), o modelo fine-tunado é carregado e usado para inferência via `pipeline`.

---

## `datasets`

Usado em: `genai_aula3_3_models_transformer.ipynb`

Biblioteca da HuggingFace para manipulação eficiente de datasets. O DataFrame pandas é convertido em um objeto `Dataset` do HuggingFace para que o `Trainer` consiga aplicar o tokenizador e fazer o treinamento em batches de forma otimizada.

---

## Resumo por notebook


| Biblioteca      | LSTM (`aula3_1`) | FCNN (`aula3_2`) | Transformer (`aula3_3`) | Avaliação (`aula3_4`) |
| --------------- | ---------------- | ---------------- | ----------------------- | --------------------- |
| `torch`         |                  |                  | ✓                       | ✓                     |
| `matplotlib`    | ✓                | ✓                | ✓                       | ✓                     |
| `pandas`        | ✓                | ✓                | ✓                       | ✓                     |
| `scikit-learn`  | ✓                | ✓                | ✓                       | ✓                     |
| `seaborn`       | ✓                | ✓                | ✓                       | ✓                     |
| `spacy`         | ✓                | ✓                |                         | ✓                     |
| `tensorflow`    | ✓                | ✓                |                         | ✓                     |
| `tf_keras`      | ✓                | ✓                |                         | ✓                     |
| `transformers`  |                  |                  | ✓                       | ✓                     |
| `datasets`      |                  |                  | ✓                       |                       |
| `openai`        |                  |                  |                         |                       |
| `python-dotenv` |                  |                  |                         |                       |



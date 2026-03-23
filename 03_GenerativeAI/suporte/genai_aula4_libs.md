# Bibliotecas — Aula 4: Fine-Tuning de LLMs com QLoRA

Mapeamento das bibliotecas instaladas e sua utilidade no notebook da aula 4.

---

## Notebooks cobertos

| Notebook | Tema |
|---|---|
| `genai_aula4_fine_tuning.ipynb` | Fine-tuning supervisionado (SFT) de TinyLlama-1.1B para análise de sentimentos via QLoRA |

---

## Contexto geral

A aula 4 introduz o pipeline completo de **QLoRA fine-tuning**: carregamento de um LLM pré-treinado com quantização de 4 bits, inserção de adaptadores LoRA de baixo rank, treinamento supervisionado com o `SFTTrainer`, inferência com o modelo ajustado e merge dos pesos do adaptador no modelo base. O conjunto de bibliotecas é mais extenso do que nas aulas anteriores porque o processo envolve múltiplas camadas de abstração — quantização, adaptadores PEFT, treinamento com reforço e geração de texto.

---

## `transformers`

Versão: `4.49.0` — Usado em: `genai_aula4_fine_tuning.ipynb`

Biblioteca central da HuggingFace. Fornece a interface unificada para carregar modelos pré-treinados, tokenizadores e pipelines de inferência.

### Componentes utilizados

| Componente | Papel no notebook |
|---|---|
| `AutoModelForCausalLM` | Carrega o TinyLlama-1.1B pré-treinado para geração de texto causal; aplica a `BitsAndBytesConfig` durante o carregamento para quantizar os pesos em NF4 de 4 bits |
| `AutoTokenizer` | Carrega o tokenizador correspondente ao modelo base; configurado com `pad_token = eos_token` e `padding_side = "right"` |
| `BitsAndBytesConfig` | Objeto de configuração que instrui o `from_pretrained` a aplicar quantização NF4 de 4 bits durante o carregamento do modelo |
| `pipeline` | Interface de alto nível para inferência — cria um pipeline de `text-generation` que aceita o prompt no formato `[INST]...[/INST]` e retorna o texto gerado |
| `logging` | Controla o nível de verbosidade da biblioteca; configurado como `CRITICAL` para suprimir mensagens informativas durante o treinamento |

### Fluxo de uso

```
BitsAndBytesConfig (NF4 4-bit)
        ↓
AutoModelForCausalLM.from_pretrained(repositorio_hf, quantization_config=bnb_config)
        ↓
AutoTokenizer.from_pretrained(repositorio_hf)
        ↓
[treinamento com SFTTrainer]
        ↓
pipeline("text-generation", model=modelo, tokenizer=tokenizador)
```

---

## `peft`

Versão: `0.14.0` — Usado em: `genai_aula4_fine_tuning.ipynb`

**Parameter-Efficient Fine-Tuning** — biblioteca da HuggingFace que implementa técnicas de adaptação eficiente, incluindo LoRA. Em vez de atualizar todos os bilhões de parâmetros do modelo base (que ficam **congelados**), o PEFT insere matrizes treináveis de baixo rank nas camadas de atenção.

### Componentes utilizados

| Componente | Papel |
|---|---|
| `LoraConfig` | Define a configuração do adaptador: rank `r`, fator de escala `lora_alpha`, dropout, tipo de tarefa (`CAUSAL_LM`) e camadas-alvo (por padrão `q_proj` e `v_proj`) |
| `PeftModel` | Envolve o modelo base já treinado e carrega os pesos do adaptador LoRA salvo em disco, para o passo de merge |

### Parâmetros de configuração da aula

```python
peft_config = LoraConfig(
    r=8,               # rank: dimensão das matrizes A e B
    lora_alpha=16,     # escala das atualizações (alpha/r define a taxa efetiva)
    lora_dropout=0.1,  # dropout aplicado nas camadas LoRA durante o treino
    bias="none",       # bias não é adaptado
    task_type="CAUSAL_LM",
)
```

**Matemática do LoRA:** ao invés de aprender $\Delta W \in \mathbb{R}^{d \times k}$, o adaptador aprende $B \in \mathbb{R}^{d \times r}$ e $A \in \mathbb{R}^{r \times k}$, com $r \ll \min(d, k)$. A atualização efetiva é $\Delta W = B \cdot A$, escalada por $\alpha / r$.

---

## `bitsandbytes`

Versão: `0.45.3` — Usado em: `genai_aula4_fine_tuning.ipynb`

Biblioteca de otimização de memória que viabiliza o **QLoRA** por meio de quantização de pesos em 4 bits (NF4 — Normal Float 4). Sem ela, carregar o TinyLlama em float32 exigiria ~4.4 GB de VRAM; com NF4 de 4 bits, os pesos ocupam ~0.55 GB.

### Configuração utilizada

| Parâmetro | Valor | Efeito |
|---|---|---|
| `load_in_4bit=True` | `True` | Ativa a quantização de 4 bits nos pesos do modelo |
| `bnb_4bit_quant_type` | `"nf4"` | Tipo NF4, otimizado para distribuições de pesos normais de LLMs |
| `bnb_4bit_compute_dtype` | `torch.float16` | Os cálculos internos são feitos em float16, não em int4 — preserva precisão numérica |
| `bnb_4bit_use_double_quant` | `False` | Desabilita a double quantization (quantização aninhada do fator de escala) |

**Importante:** a quantização NF4 é aplicada apenas durante o **carregamento** do modelo. Durante o forward pass, os pesos são dequantizados para `float16` para o cálculo — daí o parâmetro `bnb_4bit_compute_dtype`.

---

## `trl`

Versão: `0.15.2` — Usado em: `genai_aula4_fine_tuning.ipynb`

**Transformer Reinforcement Learning** — biblioteca da HuggingFace que implementa o loop de treinamento de LLMs, incluindo SFT, RLHF e DPO. Nesta aula, usamos apenas a parte de **Supervised Fine-Tuning**.

### Componentes utilizados

| Componente | Papel |
|---|---|
| `SFTConfig` | Concentra todos os hiperparâmetros do treinamento: batch size, learning rate, otimizador, scheduler, comprimento de sequência, agrupamento por comprimento, etc. |
| `SFTTrainer` | Orquestra o loop de treinamento; integra automaticamente com `peft_config` para inserir e treinar os adaptadores LoRA, e com o dataset no formato HuggingFace |

### Parâmetros críticos para 10 GB de VRAM

```python
sft_config = SFTConfig(
    per_device_train_batch_size=1,   # reduzido para economizar memória de ativação
    gradient_accumulation_steps=8,   # batch efetivo = 1 × 8 = 8
    max_seq_length=512,              # ativações crescem quadraticamente com o comprimento
    fp16=True,                       # precisão mista — reduz uso de memória
    optim="adamw_8bit",              # otimizador AdamW quantizado em 8 bits
    gradient_checkpointing=True,     # recomputa ativações no backward — troca velocidade por VRAM
    ...
)
```

**Por que `gradient_checkpointing`?** Durante o backpropagation, o PyTorch normalmente mantém todas as ativações intermediárias em memória para calcular os gradientes. O gradient checkpointing descarta essas ativações e as recomputa sob demanda — reduz o pico de VRAM em ~60%, ao custo de ~30% mais tempo de treino.

---

## `datasets`

Versão: `3.3.2` — Usado em: `genai_aula4_fine_tuning.ipynb`

Biblioteca da HuggingFace para carregamento e manipulação eficiente de datasets. Substitui o carregamento manual com pandas para datasets maiores, pois suporta lazy loading, streaming e integração direta com o `SFTTrainer`.

### Uso no notebook

```python
dataset_carregado = load_dataset("csv", data_files="dataset/dataset.csv", delimiter=",")
# resultado: DatasetDict com split "train"
trainer = SFTTrainer(train_dataset=dataset_carregado["train"], ...)
```

O `SFTTrainer` espera um objeto `Dataset` (não um DataFrame pandas) para iterar os batches durante o treinamento. O `load_dataset` retorna diretamente esse formato, com tokenização sob demanda.

---

## `accelerate`

Versão: `1.4.0` — Usado em: `genai_aula4_fine_tuning.ipynb` (dependência indireta)

Backend de aceleração de treinamento da HuggingFace. Não é chamado diretamente no código, mas é requerido pelo `transformers` e pelo `trl` para:

- Distribuir o modelo entre dispositivos com `device_map="auto"`
- Gerenciar a precisão mista (`fp16`)
- Suportar `gradient_checkpointing` de forma compatível com PEFT

---

## `sentencepiece`

Versão: `0.2.0` — Usado em: `genai_aula4_fine_tuning.ipynb` (dependência indireta)

Biblioteca de tokenização subword usada pelo tokenizador do TinyLlama/Llama. O `AutoTokenizer` requer `sentencepiece` para decodificar o vocabulário BPE (Byte-Pair Encoding) do modelo. Sem ela, o `from_pretrained` do tokenizador lança um erro de dependência.

---

## `tokenizers`

Versão: `0.21.0` — Usado em: `genai_aula4_fine_tuning.ipynb` (dependência indireta)

Backend de alta performance (escrito em Rust) que alimenta os tokenizadores rápidos da HuggingFace (`AutoTokenizer` retorna um "fast tokenizer" por padrão). Responsável pela tokenização em batch durante o pré-processamento do dataset no `SFTTrainer`.

---

## `protobuf`

Versão: `5.29.3` — Dependência indireta

Protocol Buffers — formato de serialização da Google usado internamente pelo `sentencepiece` para carregar os arquivos de vocabulário (`.model`) dos tokenizadores. Não é referenciado diretamente no código do notebook.

---

## `scipy`

Versão: `1.13.1` — Dependência indireta

Biblioteca de computação científica requerida pelo `bitsandbytes` para algumas operações numéricas internas durante a quantização. Não é importada diretamente no notebook.

---

## `pandas`

Sem versão fixada (herdada do `requirements.txt`) — Usado em: `genai_aula4_fine_tuning.ipynb`

Utilizado exclusivamente na etapa de inspeção do dataset: converte o `Dataset` HuggingFace em um DataFrame para exibir as primeiras amostras e verificar estrutura e colunas antes do treinamento.

```python
df_preview = pd.DataFrame(dataset_carregado["train"])
df_preview.head(3)
```

---

## `torch`

Sem versão fixada (herdada do `requirements.txt`) — Usado em: `genai_aula4_fine_tuning.ipynb`

Infraestrutura de tensores sobre a qual toda a pilha HuggingFace opera. No notebook da aula 4, o uso direto é mínimo — limitado a:

| Uso | Finalidade |
|---|---|
| `torch.cuda.is_available()` | Verifica a presença de GPU antes do treinamento |
| `torch.cuda.get_device_name/Properties` | Exibe informações da GPU disponível |
| `torch.cuda.empty_cache()` | Libera memória da GPU após deletar modelo e pipeline |
| `getattr(torch, "float16")` | Resolve `torch.float16` a partir da string de configuração |
| `torch.float16` | Dtype para recarregar o modelo base no passo de merge |
| `gc.collect()` | Complementa a liberação de memória (biblioteca padrão Python, não PyTorch) |

---

## Resumo por componente

| Biblioteca | Versão | Papel principal | Uso direto |
|---|---|---|---|
| `transformers` | 4.49.0 | Carregamento do modelo, tokenizador e pipeline de inferência | ✓ |
| `peft` | 0.14.0 | Adaptadores LoRA — define e aplica os pesos treináveis | ✓ |
| `bitsandbytes` | 0.45.3 | Quantização NF4 de 4 bits — viabiliza o QLoRA em 10 GB | ✓ |
| `trl` | 0.15.2 | Loop de fine-tuning supervisionado (`SFTTrainer`) | ✓ |
| `datasets` | 3.3.2 | Carregamento do dataset CSV no formato HuggingFace | ✓ |
| `torch` | — | Infraestrutura de tensores; verificação e liberação de GPU | ✓ |
| `pandas` | — | Inspeção visual do dataset antes do treinamento | ✓ |
| `accelerate` | 1.4.0 | Backend de distribuição de dispositivos e precisão mista | indireta |
| `sentencepiece` | 0.2.0 | Tokenização BPE do Llama/TinyLlama | indireta |
| `tokenizers` | 0.21.0 | Backend Rust dos tokenizadores rápidos HuggingFace | indireta |
| `protobuf` | 5.29.3 | Serialização do vocabulário do tokenizador | indireta |
| `scipy` | 1.13.1 | Operações numéricas internas do `bitsandbytes` | indireta |

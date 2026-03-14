# Bibliotecas — Aula 4: Consumo de LLMs via API

Mapeamento das bibliotecas em `requirements.txt` e sua utilidade no notebook da aula 4.

---

## Notebooks cobertos

| Notebook | Tema |
|---|---|
| `genai_aula4_llm_api.ipynb` | Consumo do Chat Completions da OpenAI, parâmetros de geração e boas práticas |

---

## Contexto geral

A aula 4 representa uma mudança de perspectiva em relação às aulas anteriores: em vez de construir ou treinar modelos, consumimos um LLM já treinado por meio de uma API REST. Todo o poder do GPT-4o fica acessível com poucas linhas de código — o foco passa a ser **como controlar o comportamento do modelo** por meio de parâmetros e boas práticas de engenharia de prompt.

---

## `openai`

Usado em: `genai_aula4_llm_api.ipynb` — cliente principal da aula

Biblioteca oficial da OpenAI para Python. Fornece o cliente `OpenAI` e encapsula todas as requisições HTTP à API, incluindo autenticação, serialização de JSON e tratamento de respostas.

### Criação do cliente

```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### Chat Completions — `client.chat.completions.create`

O método central da aula. Recebe a lista de mensagens e os parâmetros de geração; retorna um objeto `ChatCompletion` com a resposta e metadados de uso.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    temperature=0.7,
    max_tokens=200
)
```

**Campos do objeto de resposta:**

| Campo | Descrição |
|---|---|
| `response.choices[0].message.content` | Texto gerado pelo modelo |
| `response.choices[0].finish_reason` | Motivo de parada: `stop`, `length`, `content_filter` |
| `response.usage.prompt_tokens` | Tokens consumidos pela entrada (messages) |
| `response.usage.completion_tokens` | Tokens gerados na saída |
| `response.usage.total_tokens` | Soma dos dois anteriores — base do custo |
| `response.model` | Modelo efetivamente usado (pode diferir do solicitado em aliases) |
| `response.system_fingerprint` | Identifica a versão da infraestrutura — útil com `seed` |

### Streaming — `stream=True`

Quando ativado, o método retorna um iterador de `chunks`. Cada chunk contém um fragmento do texto em `chunk.choices[0].delta.content`. Utilizado para exibir respostas progressivamente em interfaces conversacionais.

### Tratamento de erros

A biblioteca expõe classes de exceção específicas para cada tipo de erro:

| Exceção | Causa |
|---|---|
| `RateLimitError` | Limite de requisições por minuto ou tokens por minuto atingido |
| `AuthenticationError` | Chave de API ausente ou inválida |
| `BadRequestError` | Parâmetros inválidos ou conteúdo bloqueado por política |
| `APIStatusError` | Erro genérico do servidor (5xx) |

### Saída estruturada — `response_format`

Ao passar `response_format={"type": "json_object"}`, o modelo garante que a saída seja um JSON válido — necessário combinado com instrução no system prompt.

---

## `python-dotenv`

Usado em: `genai_aula4_llm_api.ipynb` — carregamento seguro de credenciais

Carrega a variável `OPENAI_API_KEY` do arquivo `.env` sem expô-la no código ou no repositório. É a primeira chamada do notebook, antes de qualquer acesso à API.

```python
from dotenv import load_dotenv
load_dotenv(override=True)
```

O parâmetro `override=True` garante que os valores do `.env` sobrescrevam variáveis de ambiente já definidas no sistema operacional — útil em ambientes de desenvolvimento onde múltiplas chaves coexistem.

---

## Resumo por notebook

| Biblioteca      | `aula4_llm_api` |
|-----------------|:---------------:|
| `openai`        | ✓               |
| `python-dotenv` | ✓               |

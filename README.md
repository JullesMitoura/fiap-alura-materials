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
<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg"
     alt="HuggingFace"
     width="48"
     height="48"/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)

<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>

</div>

# Inteligência Artificial — Materiais de Aula
> PhD. Julles Mitoura

Repositório com os materiais práticos das disciplinas de IA. A trilha percorre uma progressão intencional: dos fundamentos matemáticos de aprendizado de máquina até a construção e consumo de modelos de linguagem em produção.

---

## Módulos

| # | Módulo | Tema Central | Frameworks |
|---|--------|--------------|------------|
| 01 | [Machine Learning](./01_MachineLearning/) | Regressão, minimização de SSE, equação normal | NumPy, Pandas |
| 02 | [Redes Neurais](./02_RedesNeurais/) | Redes neurais do zero, deep learning com PyTorch | NumPy, PyTorch |
| 03 | [IA Generativa](./03_GenerativeAI/) | Transformers, NLP, fine-tuning, LLMs via API | PyTorch, TensorFlow, HuggingFace, OpenAI |

---

## Trilha de Aprendizado

```
01 Machine Learning
   └─ Fundamentos matemáticos: álgebra linear, regressão, otimização
      └─ Dataset real: trocador de calor (engenharia)

02 Redes Neurais
   └─ Do neurônio à rede multicamada — implementação manual com NumPy
      └─ PyTorch: autograd, DataLoader, GPU
         └─ LSTM: memória, gates, séries temporais

03 IA Generativa
   └─ Transformer do zero (NumPy) e com PyTorch
      └─ NLP: LSTM bidirecional, FCNN+TF-IDF, fine-tuning BERT
         └─ LLMs em produção: API OpenAI, parâmetros, boas práticas
```

A progressão é intencional: cada módulo constrói sobre o anterior. Ao chegar nos Transformers, os fundamentos de gradiente descendente, backpropagation e redes recorrentes já são familiares.

---

## Como começar

Cada módulo possui seu próprio `requirements.txt`. Instale as dependências dentro de um ambiente virtual:

```bash
cd 01_MachineLearning   # ou 02_RedesNeurais, 03_GenerativeAI
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
jupyter notebook
```

Abra o notebook desejado e execute as células em sequência (`Run All`). Cada notebook é independente e autocontido.

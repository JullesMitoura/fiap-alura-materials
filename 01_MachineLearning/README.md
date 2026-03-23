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
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg"
     alt="NumPy"
     width="48"
     height="48"/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)


<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>

</div>

# Machine Learning — Fundamentos e Modelagem Preditiva
> PhD. Julles Mitoura

Módulo introdutório que conduz o aluno dos fundamentos matemáticos de aprendizado supervisionado — álgebra linear, otimização e minimização de erro — até a aplicação em um dataset de engenharia real. Ponto de partida antes de entrar em redes neurais.

---

## Estrutura do Módulo

| Aula | Notebook | Tema | Conteúdo Principal |
|------|----------|------|--------------------|
| 00 | [aula0_machine_learning.ipynb](./aula0_machine_learning.ipynb) | Regressão Linear e Minimização do SSE | Equação normal, derivação matricial de β, gradiente descendente, aplicação em dados reais de trocador de calor |

---

## Progressão Pedagógica

```
Aula 00 → Modelagem supervisionada: da função de custo (SSE) à solução fechada (equação normal)
          → Implementação manual com NumPy: sem frameworks, sem abstrações
          → Validação em dataset real de engenharia
```

O módulo é intencionalmente autocontido: antes de qualquer framework ou rede neural, o aluno implementa o ciclo completo de aprendizado — definição do modelo, função de custo, otimização e avaliação.

---

## Datasets

| Aula | Dataset | Descrição |
|------|---------|-----------|
| 00 | `data/heat_exchanger.csv` | Dados reais de trocador de calor (engenharia) para regressão supervisionada |

---

## Material de Suporte

| Arquivo | Descrição |
|---------|-----------|
| [`suporte/aula0_min_sse.md`](./suporte/aula0_min_sse.md) | Derivação completa da equação normal: do SSE matricial à solução β = (XᵀX)⁻¹Xᵀy |

---

## Pré-requisitos

```bash
pip install -r requirements.txt
```

| Biblioteca | Versão mínima | Uso |
|------------|--------------|-----|
| `numpy` | 1.20.0 | Álgebra linear, operações matriciais, gradiente descendente |
| `matplotlib` | 3.5.0 | Visualizações e gráficos de regressão |
| `pandas` | 1.3.0 | Leitura e manipulação do dataset CSV |

---

## Como executar

Abra o notebook no Jupyter e execute todas as células em sequência (`Run All`). O notebook é independente e autocontido.

```bash
jupyter notebook
```

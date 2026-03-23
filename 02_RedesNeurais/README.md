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

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)


<img width="80%" src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif"/>

</div>

# Redes Neurais Artificiais, Deep Learning e Algoritmos Genéticos
> PhD. Julles Mitoura

Módulo prático que conduz o aluno desde os fundamentos matemáticos de uma rede neural até a implementação com PyTorch — construindo cada componente do zero antes de utilizar frameworks, para que a abstração faça sentido.

---

## Estrutura do Módulo

| Aula | Notebook | Tema | Conteúdo Principal |
|------|----------|------|--------------------|
| 00 | [aula0_redes_neurais.ipynb](./aula0_redes_neurais.ipynb) | Regressão Linear com Gradiente Descendente | Forward pass, MSE, backpropagation manual, loop de treino com NumPy |
| 01 | [aula1_redes_neurais.ipynb](./aula1_redes_neurais.ipynb) | O que é uma Rede Neural? | Neurônio artificial, redes multicamada do zero, normalização Z-score, aplicação em regressão real |
| 03 | [aula3_redes_neurais_fnn.ipynb](./aula3_redes_neurais_fnn.ipynb) | Fully Connected NN com PyTorch | `nn.Module`, autograd, Adam, regressão com CSV real e classificação multiclasse (bullseye) |
| 04 | [aula4_redes_neurais_lstm.ipynb](./aula4_redes_neurais_lstm.ipynb) | LSTM com PyTorch | Gates, estado de célula, janela deslizante, previsão de série temporal e classificação de sequências |

---

## Progressão Pedagógica

```
Aula 00 → Um neurônio, gradiente descendente na forma mais simples
Aula 01 → Rede multicamada completa: forward, loss, backward, update (só NumPy)
Aula 03 → Mesma rede, agora com PyTorch: autograd, DataLoader, treinamento com GPU
Aula 04 → LSTM: memória, gates e aprendizado de dependências temporais
```

A progressão é intencional: você implementa tudo manualmente antes de usar o framework. Quando o PyTorch é introduzido, cada componente já é familiar.

---

## Datasets

| Aula | Dataset | Descrição |
|------|---------|-----------|
| 00 | Sintético | Relação linear $y = wx + b + \varepsilon$ com ruído gaussiano |
| 01 | Sintético + CSV | Dataset gerado e `data/regression_example.csv` (400 amostras, 4 features) |
| 03 | CSV + Sintético | `data/regression_example.csv` para regressão; bullseye sintético (3 classes por faixa de raio) para classificação |
| 04 | Sintético | Série temporal $\sin(t) + 0.5\sin(3t) + \varepsilon$ para regressão; sequências senoidais, lineares e ruído para classificação |

---

## Pré-requisitos

```bash
pip install -r requirements.txt
```

| Biblioteca | Versão mínima | Uso |
|------------|--------------|-----|
| `numpy` | 1.20.0 | Álgebra linear, geração de datasets |
| `matplotlib` | 3.5.0 | Visualizações e gráficos |
| `pandas` | 1.3.0 | Leitura de CSV (aulas 01 e 03) |
| `scikit-learn` | 1.0 | Normalização com `StandardScaler` (aula 04) |
| `torch` | — | Framework de deep learning (aulas 03 e 04) |

---

## Como executar

Abra qualquer notebook no Jupyter e execute todas as células em sequência (`Run All`). Cada notebook é independente e autocontido.

```bash
jupyter notebook
```

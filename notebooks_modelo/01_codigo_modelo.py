# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %% [markdown]
# ### IMPORTAÇÃO DAS BIBLIOTECAS

# %%
#Instalando o pacote watermark
# !pip install -q -U watermark

# %%
#Manipulação e visualização de dados
import pandas as pd
import re
import numpy as np
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt

#Pré-processamento e machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Deploy
import joblib

# %%
#Configurações de visualização
sns.set_style('whitegrid')
# %matplotlib inline

# %%
# %reload_ext watermark
# %watermark -a "Davi de Andrade Leandro"

# %%
# %watermark --iversions

# %% [markdown]
# ### CARREGANDO OS DADOS

# %%
df = pd.read_csv(r'C:\Users\DaviAndrade\Downloads\Projetos\Projetos Ciência de dados\analise_de_sentimentos\data\dataset.csv')

df.head()

# %%
#Amostra aleatoria
df.sample(5)

# %%
df.shape

# %% [markdown]
# ### EDA

# %%
df.info()

# %%
print('Usando o método info(),verificamos valores ausentes na coluna "texto_review",agora vamos quantificar.\n')
print('    VALORES NULOS\n')

display(df.isnull().sum())

# %%
print('Para facilitar a nossa visualização, vamos identificar a proporção de classes.\n')

sns.countplot(x='sentimento',data=df,color='gray')
plt.title('Distribuição de classes de sentimentos')
plt.show()

print('\nPodemos ver que os nossos dados são bem balanceados!')

# %% [markdown]
# ### LIMPEZA DE DADOS

# %%
#Vamos apagar os 12 registros ausentes

print(f'\nTamanho do dataset antes de excluir os dados ausentes: {len(df)}')
df.dropna(subset=['texto_review'],inplace=True)
print(f'Tamanho do dataset após excluir os dados ausentes: {len(df)}')


# %% [markdown]
# #### IMPORTANTE: Podemos identificar também, valores irregulares na coluna "texto_review", como acentos, erros de digitação e outros.Vamos aplicar uma função para corrigir todos esses erros 

# %%
# Função de limpeza de texto 
def limpa_texto(texto):
    
    """
    Função completa de limpeza de texto:
    1. Converte para minúsculas.
    2. Remove acentos e cedilha.
    3. Remove pontuações, números e caracteres especiais.
    4. Remove espaços extras.
    """
    
    # Garante que o texto não seja nulo (caso haja algum NaN no DataFrame)
    if not isinstance(texto, str):
        return ""

    # --- PASSO 1: Normalizar e remover acentos ---
    # Normaliza para a forma 'NFKD' que separa o caractere da acentuação
    # e depois remove os acentos (Nonspacing Mark)
    texto_sem_acentos = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')

    # --- PASSO 2: Limpeza com Regex ---
    # Converter para minúsculas
    texto_limpo = texto_sem_acentos.lower()
    
    # Manter apenas letras e espaços. A remoção de acentos já foi feita.
    texto_limpo = re.sub(r'[^a-z\s]', '', texto_limpo)
    
    # Remover espaços extras
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    
    return texto_limpo


# %%
#Vamos aplicar a função
df['Texto_limpo'] = df['texto_review'].apply(limpa_texto)

df.head()

# %% [markdown]
# ### ENGENHARIA DE ATRIBUTOS

# %%
#Vamos criar uma classificação para a coluna sentimento
df['sentimento_label'] = df['sentimento'].map({'negativo':0,'positivo':1})

# %%
#Agora excluir as colunas redudantes
df = df.drop(columns=['texto_review','sentimento'],axis=1)

# %%
df.head()

# %% [markdown]
# ### SEPARAÇÃO EM TREINO E TESTE

# %%
X = df['Texto_limpo']
y = df['sentimento_label']

# %%
X_train,X_test,y_train,y_test = train_test_split(
    X, y , test_size= 0.25 , random_state=42, stratify= y
)

# %% [markdown]
# ### PIPELINE DE MODELAGEM PREDITIVA
# Aqui, construímos uma esteira de produção automatizada para o nosso modelo. O Pipeline do Scikit-learn encapsula todas as etapas de pré-processamento (vetorização TF-IDF, padronização com StandardScaler) e o modelo final (Regressão Logística), garantindo que os mesmos passos sejam aplicados de forma consistente nos dados de treino e nos novos dados.

# %%
#pipeline

pipeline = Pipeline([
    ('tfidf',TfidfVectorizer(stop_words=['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um'])), #Converendo as palavras em números
    ('scaler',StandardScaler(with_mean=False)), #Ajustando a escala
    ('logreg',LogisticRegression(solver='liblinear',random_state=42, max_iter=1000)) #Criando o algoritmo
    
])


# %%
type(pipeline)

# %% [markdown]
# O pipeline é composto por três etapas sequenciais, cada uma com um nome ('tfidf', 'scaler', 'logreg') e uma função específica.

# %% [markdown]
# ### Otimização de Hiperparâmetros
#
# É o ajuste fino do modelo. Usando GridSearchCV, testamos sistematicamente várias combinações de configurações (hiperparâmetros) para o pipeline, a fim de encontrar a combinação que resulta na melhor performance possível.

# %%
# Definir o grid de hiperparâmetros para otimização
parametros_grid = {
    'tfidf__max_features': [500, 1000, 2000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'logreg__C': [0.1, 1, 10],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__max_iter': [5000, 6000]
}

# %%
# Configurar o GridSearchCV (Validação Cruzada)
grid_search = GridSearchCV(
    pipeline,              # Pipeline com as etapas de pré-processamento e modelo
    parametros_grid,       # Dicionário com as combinações de hiperparâmetros a serem testadas
    cv = 5,                # Número de divisões para validação cruzada (5-fold cross-validation)
    n_jobs = -1,           # Usa todos os núcleos disponíveis do processador para acelerar o processo
    scoring = 'accuracy',  # Métrica usada para avaliar o desempenho de cada combinação (aqui, acurácia)
    verbose = 1            # Nível de detalhamento do output durante a execução (1 exibe progresso básico)
)

# %% [markdown]
# Validação cruzada é uma técnica usada para avaliar o desempenho de um modelo dividindo o conjunto de dados em várias partes (ou “folds”). O modelo é treinado em algumas dessas partes e testado em outras, de forma rotativa. Isso permite medir o desempenho de forma mais confiável e geral, evitando que o resultado dependa apenas de uma única divisão dos dados.

# %% [markdown]
# ### TREINAMENTO DO MODELO

# %%
print('\nIniciando o treinamento do modelo com otimização de hiperparâmetros\n')
grid_search.fit(X_train,y_train)

# %%
print('\nMelhores hiperparâmetros do encontrados\n')
print(grid_search.best_params_)

# %%
#Vamos salvar os melhores hiperparâmetros em uma variável
melhor_modelo = grid_search.best_estimator_

# %%
type(melhor_modelo)

# %% [markdown]
# ### Avaliação do Modelo e Interpretação de Métricas
#
# É a "prova final". Usamos o conjunto de teste (os dados que o modelo nunca viu) para fazer previsões e compará-las com os resultados reais. Métricas como Acurácia, Relatório de Classificação e a Matriz de Confusão nos dizem quão bem o modelo está generalizando e se ele atende aos objetivos de negócio.

# %%
# Previsões no conjunto de teste
y_pred = melhor_modelo.predict(X_test)

# %%
# Calcular as métricas de avaliação
acuracia = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names = ['Negativo', 'Positivo'])

# %%
print(f"\nAcurácia do Modelo: {acuracia:.2%}\n")
print("Relatório de Classificação:\n")
print(report)

# %%
# Visualizar a Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels = ['Negativo', 'Positivo'],
            yticklabels = ['Negativo', 'Positivo'])
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# %% [markdown]
# ### DEPLOY DO MODELO

# %%
# Se estivermos satisfeitos com a performance do modelo, salvamos em disco
joblib.dump(melhor_modelo, 'modelo_sentimento_v1.joblib')

# %%
# Pode deletar o modelo treinado e removê-lo da memória
del melhor_modelo

# %%
# Carregar o modelo a partir do disco
modelo_deploy = joblib.load('modelo_sentimento_v1.joblib')

# %%
type(modelo_deploy)

# %%
# Criar novos dados para simular o uso em produção
novos_reviews = [
    "A bateria do celular não dura nada, péssima compra.",
    "Chegou antes do prazo e o produto é de ótima qualidade! Estou muito feliz.",
    "O serviço de atendimento foi rápido e eficiente.",
    "Não recomendo, veio faltando peças e a cor estava errada."
]


# %%
# Função para prever o sentimento de novos reviews 
def prever_sentimento(reviews):
    
    """
    Recebe uma lista de textos de review e retorna a previsão de sentimento.
    O objeto 'melhor_modelo_dsa' (pipeline) cuida de todos os passos internos.
    """
    
    # 1. 'reviews' entra no pipeline
    # 2. TF-IDF é aplicado internamente
    # 3. StandardScaler é aplicado internamente
    # 4. LogisticRegression faz a previsão
    previsoes = modelo_deploy.predict(reviews)
    
    # Mapeia o resultado numérico de volta para texto
    sentimentos = ['Negativo' if p == 0 else 'Positivo' for p in previsoes]
    
    # Exibe os resultados
    for review, sentimento in zip(reviews, sentimentos):
        print(f"\nReview: '{review}'\nSentimento Previsto: {sentimento}\n---")


# %%
# Executar a função de deploy com os novos dados
print("\n--- Iniciando Classificação de Novos Reviews (Deploy com Pipeline Completo) ---\n")
prever_sentimento(novos_reviews)

# %%

# %%

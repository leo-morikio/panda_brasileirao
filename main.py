import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

# --- Configuração Inicial ---
sns.set(style="whitegrid")

# --- Passo 1: Carregar os dados do Kaggle diretamente ---
file_path = "campeonato-brasileiro-full.csv"

try:
   df = kagglehub.load_dataset(
       KaggleDatasetAdapter.PANDAS,
       "adaoduque/campeonato-brasileiro-de-futebol",
       file_path,
   )
   print(f"Dados carregados com sucesso do Kaggle: {file_path}")
except Exception as e:
   print(f"Erro ao carregar dataset do Kaggle: {e}")
   exit()

df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# --- Passo 2: Preparação dos Dados ---
colunas_para_remover = ['formacao_mandante', 'formacao_visitante', 'tecnico_mandante', 'tecnico_visitante']
df = df.drop(columns=[col for col in colunas_para_remover if col in df.columns])

# --- Passo 3: Análise Descritiva ---
tipo_vitoria = []
for i in df.index:
   if df.at[i, 'vencedor'] == df.at[i, 'mandante']:
       tipo_vitoria.append('Mandante')
   elif df.at[i, 'vencedor'] == df.at[i, 'visitante']:
       tipo_vitoria.append('Visitante')
   else:
       tipo_vitoria.append('Empate')
df['tipo_vitoria'] = tipo_vitoria

contagem_resultado = df['tipo_vitoria'].value_counts()
vitorias_mandante = contagem_resultado.get('Mandante', 0)
vitorias_visitante = contagem_resultado.get('Visitante', 0)
empates = contagem_resultado.get('Empate', 0)

# --- Passo 4: Visualizações ---

# Histograma dos gols marcados pelo time mandante
plt.figure(figsize=(9, 6))
sns.histplot(df['mandante_placar'], bins=range(df['mandante_placar'].min(), df['mandante_placar'].max() + 2), kde=False, discrete=True)
plt.title('Distribuição de Gols do Time Mandante')
plt.xlabel('Número de Gols')
plt.ylabel('Frequência de Jogos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Histograma dos gols marcados pelo time visitante
plt.figure(figsize=(9, 6))
sns.histplot(df['visitante_placar'], bins=range(df['visitante_placar'].min(), df['visitante_placar'].max() + 2), kde=False, color='orange', discrete=True)
plt.title('Distribuição de Gols do Time Visitante')
plt.xlabel('Número de Gols')
plt.ylabel('Frequência de Jogos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Heatmap da frequência dos placares combinados (mandante x visitante)
placares = {}
for i in df.index:
   m = df.at[i, 'mandante_placar']
   v = df.at[i, 'visitante_placar']
   if (m, v) not in placares:
       placares[(m, v)] = 1
   else:
       placares[(m, v)] += 1

max_m = df['mandante_placar'].max()
max_v = df['visitante_placar'].max()
matriz = np.zeros((max_m + 1, max_v + 1), dtype=int)
for (m, v), qtd in placares.items():
   matriz[m][v] = qtd

plt.figure(figsize=(12, 8))
sns.heatmap(matriz, annot=True, fmt='d', cmap='YlOrBr', linewidths=0.5, linecolor='gray')
plt.title('Frequência de Placares: Gols do Mandante vs Gols do Visitante')
plt.xlabel('Gols do Visitante')
plt.ylabel('Gols do Mandante')
plt.tight_layout()
plt.show()

# Boxplot comparando a distribuição dos gols entre mandantes e visitantes
tipo = ['Mandante'] * len(df) + ['Visitante'] * len(df)
placar = df['mandante_placar'].tolist() + df['visitante_placar'].tolist()
df_box = pd.DataFrame({'tipo': tipo, 'placar': placar})

plt.figure(figsize=(9, 6))
sns.boxplot(x='tipo', y='placar', data=df_box, palette=['skyblue', 'lightcoral'])
plt.title('Distribuição de Gols por Tipo de Time')
plt.xlabel('Tipo de Time')
plt.ylabel('Número de Gols')
plt.tight_layout()
plt.show()

# Gráfico de barras mostrando o total de vitórias por tipo e empates
plt.figure(figsize=(8, 6))
sns.barplot(x=['Mandante', 'Visitante', 'Empate'], y=[vitorias_mandante, vitorias_visitante, empates], palette=['blue', 'orange', 'gray'])
plt.title('Total de Resultados: Mandante vs Visitante vs Empate')
plt.ylabel('Número de Jogos')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Pizza mostrando a proporção percentual dos resultados (vitória mandante, empate, vitória visitante)
plt.figure(figsize=(8, 8))
cores = ['skyblue', 'lightgray', 'orange']
plt.pie([vitorias_mandante, empates, vitorias_visitante], labels=['Vitória Mandante', 'Empate', 'Vitória Visitante'], autopct='%1.1f%%', startangle=90, colors=cores, wedgeprops={'edgecolor': 'black'})
plt.title('Proporção dos Resultados dos Jogos')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Gráfico de linha mostrando a média de gols por rodada para mandantes e visitantes
rodadas = sorted(df['rodata'].dropna().unique())
gols_mandante_por_rodada = []
gols_visitante_por_rodada = []
for r in rodadas:
   dados_rodada = df[df['rodata'] == r]
   gols_mandante_por_rodada.append(dados_rodada['mandante_placar'].mean())
   gols_visitante_por_rodada.append(dados_rodada['visitante_placar'].mean())

plt.figure(figsize=(15, 8))
plt.plot(rodadas, gols_mandante_por_rodada, marker='o', label='Mandante')
plt.plot(rodadas, gols_visitante_por_rodada, marker='o', label='Visitante')
plt.title('Média de Gols por Rodada: Mandante vs. Visitante')
plt.xlabel('Rodada')
plt.ylabel('Média de Gols')
plt.legend(title='Tipo de Time')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Heatmap mostrando o total de gols por rodada em cada temporada
df['temporada'] = pd.to_datetime(df['data'], dayfirst=True, errors='coerce').dt.year
df['rodata'] = df['rodata'].astype(str).str.extract(r'(\d+)').astype(float)
df['gols_totais'] = df['mandante_placar'] + df['visitante_placar']

gols_por_rodada = {}
for i in df.index:
   ano = df.at[i, 'temporada']
   rodada = df.at[i, 'rodata']
   if pd.isnull(ano) or pd.isnull(rodada):
       continue
   chave = (ano, rodada)
   if chave not in gols_por_rodada:
       gols_por_rodada[chave] = df.at[i, 'gols_totais']
   else:
       gols_por_rodada[chave] += df.at[i, 'gols_totais']

temporadas = sorted(set([k[0] for k in gols_por_rodada.keys()]))
rodadas = sorted(set([k[1] for k in gols_por_rodada.keys()]))
heatmap_matrix = []
for ano in temporadas:
   linha = []
   for rodada in rodadas:
       linha.append(gols_por_rodada.get((ano, rodada), 0))
   heatmap_matrix.append(linha)

plt.figure(figsize=(16, 8))
sns.heatmap(heatmap_matrix, cmap='YlOrBr', linewidths=0.5, linecolor='gray', annot=False)
plt.title('Total de Gols por Rodada em Cada Temporada')
plt.xlabel('Rodada')
plt.ylabel('Temporada')
plt.yticks(ticks=np.arange(len(temporadas)) + 0.5, labels=temporadas, rotation=0)
plt.xticks(ticks=np.arange(len(rodadas)) + 0.5, labels=[int(r) for r in rodadas], rotation=90)
plt.tight_layout()
plt.show()

# Análise da distribuição dos resultados na temporada 2020
df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
df_2020 = df[df['data'].dt.year == 2020]
total_2020 = len(df_2020)

tipo_vitoria_2020 = []
for i in df_2020.index:
   if df_2020.at[i, 'mandante_placar'] > df_2020.at[i, 'visitante_placar']:
       tipo_vitoria_2020.append('Mandante')
   elif df_2020.at[i, 'mandante_placar'] < df_2020.at[i, 'visitante_placar']:
       tipo_vitoria_2020.append('Visitante')
   else:
       tipo_vitoria_2020.append('Empate')
df_2020['tipo_vitoria'] = tipo_vitoria_2020

contagem_2020 = df_2020['tipo_vitoria'].value_counts()
porcentagens_2020 = round(contagem_2020 / total_2020 * 100, 2)

labels = porcentagens_2020.index
sizes = porcentagens_2020.values
colors = ['skyblue', 'lightcoral', 'lightgrey']

# Pizza mostrando a proporção dos resultados dos jogos da temporada 2020
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'black'})
plt.title('Distribuição dos Resultados em 2020')
plt.axis('equal')
plt.tight_layout()
plt.show()

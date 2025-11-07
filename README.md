# Performance-Analysis-in-Tennis
Performance Analysis in Tennis Using GPS,Heart Rate Data, Active energy (KJ) and distance.
--------------------------------
# Importar o CSV do health
import pandas as pd
from google.colab import files
import io

print("Por favor, faça o upload do arquivo CSV 'Compilado':")
uploaded_compilado = files.upload()

# Get the filename (assuming only one file is uploaded)
for fn_compilado in uploaded_compilado.keys():
  print(f'Arquivo "{fn_compilado}" carregado.')
  dados = pd.read_csv(io.StringIO(uploaded_compilado[fn_compilado].decode('utf-8'))) # Compilar manualmente deixando as colunas do lado uma da outra
dados.head(2)

#Limpeza de colunas desnecessárias
columns_to_drop = [
    'Min (count/min)',
    'Max (count/min)',
    'Context',
    'Source;Date/Time',
    'Source;Date/Time.1',
    'Source'
]

existing_columns_to_drop = [col for col in columns_to_drop if col in dados.columns]

# Drop the columns from the DataFrame
dados = dados.drop(columns=existing_columns_to_drop)

dados.head(2)

#Retirar a data do timestamp
dados['Date/Time'] = pd.to_datetime(dados['Date/Time']).dt.time
print("Coluna 'Date/Time' atualizada para mostrar apenas a hora:")
dados.head(2)


# Renomear colunas
dados.columns = ["Tempo (Min)","Frequência Cardíaca (Bpm)","Distância percorrida (Km/min)","Energia ativa (Kj/min)"]
dados.head(2)

# Importar o CSV do Rota
import pandas as pd
from google.colab import files
import io

# Read the CSV with space as delimiter and no header, using a raw string for regex
print("Por favor, faça o upload do arquivo CSV 'Rota':")
uploaded_rota = files.upload()

# Get the filename (assuming only one file is uploaded)
for fn_rota in uploaded_rota.keys():
  print(f'Arquivo "{fn_rota}" carregado.')
  rota = pd.read_csv(io.StringIO(uploaded_rota[fn_rota].decode('utf-8')), sep=r'\s+', header=None)

# Rename columns for clarity
rota.columns = ['Latitude', 'Longitude']

# Remove trailing commas from Latitude and Longitude columns and convert to numeric
rota['Latitude'] = rota['Latitude'].str.replace(',', '', regex=False).astype(float)
rota['Longitude'] = rota['Longitude'].str.replace(',', '', regex=False).astype(float)

print("DataFrame 'rota' after cleaning and conversion:")
rota.head(2)

# Como a planilha de GPS contém muito mais dados que as medições do health, é necessário igualar a quantidade de linhas,
# este código faz a media das linhas do Rota pra chegar na mesma quantidade de linhas do Compilado

import numpy as np

# Get the number of rows for both DataFrames
N_dados = len(dados)
N_rota = len(rota)

print(f"Original 'rota' rows: {N_rota}")
print(f"Target 'dados' rows: {N_dados}")

# Check if resampling is needed
if N_rota == N_dados:
    # No resampling needed if they already have the same number of rows
    rota_resampled = rota.copy()
elif N_rota < N_dados:
    # This scenario implies upsampling, which would typically involve interpolation
    # rather than averaging chunks. For this specific request, we focus on downsampling.
    # If upsampling is required in the future, a different method will be used.
    print(f"Warning: 'rota' has fewer rows ({N_rota}) than 'dados' ({N_dados}). No downsampling performed by averaging.")
    rota_resampled = rota.copy()
else: # N_rota > N_dados, perform downsampling by averaging
    # Create N_dados equally spaced bins across the index of 'rota'
    # The bins will define which rows of 'rota' get grouped together for averaging.
    bins = np.linspace(-0.5, N_rota - 0.5, N_dados + 1)

    # Reset index temporarily to ensure a contiguous integer index for pd.cut
    rota_indexed = rota.reset_index(drop=True)

    # Assign each row of 'rota' to a bin (group)
    # labels=False assigns integer labels (0 to N_dados-1) to the bins
    # include_lowest=True ensures the first interval includes the lowest value (-0.5)
    rota_indexed['group'] = pd.cut(rota_indexed.index, bins=bins, labels=False, include_lowest=True)

    # Calculate the mean for 'Latitude' and 'Longitude' for each group
    rota_resampled = rota_indexed.groupby('group')[['Latitude', 'Longitude']].mean()

    # Reset the index to make 'group' a regular column again and drop it, resulting in a clean DataFrame
    rota_resampled = rota_resampled.reset_index(drop=True)

print(f"Resampled 'rota' rows: {len(rota_resampled)}")
rota_resampled.head(2)

# Combina os dados do Rota com o Compilado
dados_completos = pd.concat([dados, rota_resampled], axis=1)

# Reorder columns to place Latitude and Longitude first
current_columns = dados_completos.columns.tolist()
new_column_order = ['Latitude', 'Longitude'] + [col for col in current_columns if col not in ['Latitude', 'Longitude']]
dados_completos = dados_completos[new_column_order]

print("Combined DataFrame 'dados_completos':")
dados_completos.head(2)

#Normalização da planilha de gps (Zera a posição inicial da marcação do relógio, PONTO ZERO)
import numpy as np

# 1. Extract the latitude and longitude of the first point in the `rota` DataFrame
ref_lat_rota = rota.loc[0, 'Latitude']
ref_lon_rota = rota.loc[0, 'Longitude']

# 2. Calculate 'x_relativo' for each point in `rota`
rota['x_relativo'] = (rota['Longitude'] - ref_lon_rota) * 111320 * np.cos(np.radians(ref_lat_rota))

# 3. Calculate 'y_relativo' for each point in `rota`
rota['y_relativo'] = (rota['Latitude'] - ref_lat_rota) * 110540

print("DataFrame 'rota' with new relative coordinates:")
print(rota.head(2))

# Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

# Normaliza a marcação dos dados completos para marcar o mapa com os pontos
ref_lat_rota = rota.loc[0, 'Latitude']
ref_lon_rota = rota.loc[0, 'Longitude']

# 2. Calculate 'x_relativo' for each point in `rota`
rota['x_relativo'] = (rota['Longitude'] - ref_lon_rota) * 111320 * np.cos(np.radians(ref_lat_rota))

# 3. Calculate 'y_relativo' for each point in `rota`
rota['y_relativo'] = (rota['Latitude'] - ref_lat_rota) * 110540
# --- END: Added code ---

# --- START: Fix for NameError: name 'df' is not defined ---
df = dados_completos.copy()
df['x_relativo'] = (df['Longitude'] - ref_lon_rota) * 111320 * np.cos(np.radians(ref_lat_rota))
df['y_relativo'] = (df['Latitude'] - ref_lat_rota) * 110540
# --- END: Fix for NameError: name 'df' is not defined ---


# Tamanho da quadra de tênis simples
COURT_LENGTH = 23.77 # Comprimento em metros
COURT_WIDTH = 8.23   # Largura em metros
SERVICE_LINE_FROM_NET = 6.40 # meters

# Baseline from net is half the court length
BASELINE_FROM_NET = COURT_LENGTH / 2 # 11.885 meters

# --- Plot Setup ---
plt.figure(figsize=(10, 15)) # Adjusted figure size to better fit the court proportions
ax = plt.gca() # Get current axes

# --- Centralizando em x0 e y0 ---
court_x_center = 0
net_y = 0 # Net is at the center of the Y-axis

# Calculate court boundaries based on fixed dimensions
court_bottom_left_x = court_x_center - (COURT_WIDTH / 2)
court_bottom_left_y = net_y - (COURT_LENGTH / 2)
court_top_right_x = court_x_center + (COURT_WIDTH / 2)
court_top_right_y = net_y + (COURT_LENGTH / 2)

# --- Draw Main Court Area ---
court_patch = patches.Rectangle((court_bottom_left_x, court_bottom_left_y), COURT_WIDTH, COURT_LENGTH,
                                edgecolor='black', facecolor='#228B22', linewidth=2, zorder=0, alpha=0.5)
ax.add_patch(court_patch)

# --- Draw Court Lines ---
# Net line
ax.plot([court_bottom_left_x, court_top_right_x],
        [net_y, net_y], color='white', linewidth=1.5, zorder=2)

# Service lines (horizontal lines at +/- SERVICE_LINE_FROM_NET from the net)
service_line_y_pos = net_y + SERVICE_LINE_FROM_NET
service_line_y_neg = net_y - SERVICE_LINE_FROM_NET
ax.plot([court_bottom_left_x, court_top_right_x],
        [service_line_y_pos, service_line_y_pos], color='white', linewidth=1, zorder=2)
ax.plot([court_bottom_left_x, court_top_right_x],
        [service_line_y_neg, service_line_y_neg], color='white', linewidth=1, zorder=2)

# Center service line (vertical line dividing the service boxes, from service line to service line)
ax.plot([court_x_center, court_x_center],
        [service_line_y_neg, service_line_y_pos], color='white', linewidth=1, zorder=2)

# --- Define and plot Custom Zones ---
# The court width for the zones is the full singles court width
zone_width = COURT_WIDTH
zone_bottom_left_x = court_bottom_left_x

# 1. Rede (Net) zone - Updated to 3.2m from center
net_zone_y_min = -3.2
net_zone_y_max = 3.2

net_zone_patch = patches.Rectangle((zone_bottom_left_x, net_zone_y_min), zone_width, net_zone_y_max - net_zone_y_min,
                                  edgecolor='blue', facecolor='cyan', linewidth=1, alpha=0.3, zorder=2, label='Rede')
ax.add_patch(net_zone_patch)

# 2. Meio (Middle) zone - Updated to 3.2m to 7m from center
# Positive Y side
middle_zone_pos_y_min = 3.2
middle_zone_pos_y_max = 7.0
middle_zone_pos_patch = patches.Rectangle((zone_bottom_left_x, middle_zone_pos_y_min), zone_width, middle_zone_pos_y_max - middle_zone_pos_y_min,
                                        edgecolor='purple', facecolor='magenta', linewidth=1, alpha=0.3, zorder=2, label='Meio (lado Positivo)')
ax.add_patch(middle_zone_pos_patch)

# Negative Y side
middle_zone_neg_y_min = -7.0
middle_zone_neg_y_max = -3.2
middle_zone_neg_patch = patches.Rectangle((zone_bottom_left_x, middle_zone_neg_y_min), zone_width, middle_zone_neg_y_max - middle_zone_neg_y_min,
                                        edgecolor='purple', facecolor='magenta', linewidth=1, alpha=0.3, zorder=2, label='Meio (lado Negativo)')
ax.add_patch(middle_zone_neg_patch)

# 3. Fundo (Back) zone - Updated to 7m to 15m from center
back_zone_extension = 5 # meters beyond the baseline (this variable is now unused for zone definitions but kept for context)

# Positive Y side
back_zone_pos_y_min = 7.0
back_zone_pos_y_max = 15.0
back_zone_pos_patch = patches.Rectangle((zone_bottom_left_x, back_zone_pos_y_min), zone_width, back_zone_pos_y_max - back_zone_pos_y_min,
                                      edgecolor='darkgreen', facecolor='lightgreen', linewidth=1, alpha=0.3, zorder=2, label='Fundo (lado Positivo)')
ax.add_patch(back_zone_pos_patch)

# Negative Y side
back_zone_neg_y_min = -15.0
back_zone_neg_y_max = -7.0
back_zone_neg_patch = patches.Rectangle((zone_bottom_left_x, back_zone_neg_y_min), zone_width, back_zone_neg_y_max - back_zone_neg_y_min,
                                      edgecolor='darkgreen', facecolor='lightgreen', linewidth=1, alpha=0.3, zorder=2, label='Fundo (lado Negativo)')
ax.add_patch(back_zone_neg_patch)


# Create the heatmap using seaborn.kdeplot with transparency
sns.kdeplot(x=rota['x_relativo'], y=rota['y_relativo'], fill=True, cmap='viridis', levels=5, cbar=True,
            cbar_kws={'label': 'Densidade de Pontos', 'shrink': 0.5}, alpha=0.7, ax=ax, zorder=3)

# Plot the normalized GPS points on top of the heatmap using the 'df' DataFrame
# Adding a check here to ensure 'x_relativo' and 'y_relativo' exist in df
if 'x_relativo' in df.columns and 'y_relativo' in df.columns:
    ax.scatter(df['x_relativo'], df['y_relativo'],
               color='red', s=50, marker='X', label='Pontos GPS Normalizados', zorder=4)
else:
    print("Warning: 'x_relativo' or 'y_relativo' not found in DataFrame 'df'. Skipping scatter plot for df.")

# Adjusted bbox_to_anchor for the legend to prevent overlap with the colorbar
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Set the plot limits to encompass all data and the simulated court, with some padding
# Adjust x-limits based on court width with some padding
ax.set_xlim(court_bottom_left_x - 1, court_top_right_x + 1)
# Adjust y-limits to cover the court length and the extended back zones with padding
ax.set_ylim(-16, 16) # Adjusted to cover the new 15m back zone with padding

plt.title('Heatmap of Relative Coordinates on Simulated Tennis Court (Fixed Dimensions & Custom Zones)')
plt.xlabel('X Relativo (metros)')
plt.ylabel('Y Relativo (metros)')
plt.grid(True, linestyle='--', alpha=0.2) # Make grid lighter

# --- Count points in each zone with new definitions ---
# Rede (Net) zone
count_net = rota[(rota['y_relativo'] >= net_zone_y_min) & (rota['y_relativo'] < net_zone_y_max) & \
                 (rota['x_relativo'] >= zone_bottom_left_x) & (rota['x_relativo'] < zone_bottom_left_x + zone_width)].shape[0]

# Meio (Middle) zone
count_middle_pos = rota[(rota['y_relativo'] >= middle_zone_pos_y_min) & (rota['y_relativo'] < middle_zone_pos_y_max) & \
                          (rota['x_relativo'] >= zone_bottom_left_x) & (rota['x_relativo'] < zone_bottom_left_x + zone_width)].shape[0]
count_middle_neg = rota[(rota['y_relativo'] >= middle_zone_neg_y_min) & (rota['y_relativo'] < middle_zone_neg_y_max) & \
                          (rota['x_relativo'] >= zone_bottom_left_x) & (rota['x_relativo'] < zone_bottom_left_x + zone_width)].shape[0]
count_middle = count_middle_pos + count_middle_neg

# Fundo (Back) zone
count_back_pos = rota[(rota['y_relativo'] >= back_zone_pos_y_min) & (rota['y_relativo'] < back_zone_pos_y_max) & \
                        (rota['x_relativo'] >= zone_bottom_left_x) & (rota['x_relativo'] < zone_bottom_left_x + zone_width)].shape[0]
count_back_neg = rota[(rota['y_relativo'] >= back_zone_neg_y_min) & (rota['y_relativo'] < back_zone_neg_y_max) & \
                        (rota['x_relativo'] >= zone_bottom_left_x) & (rota['x_relativo'] < zone_bottom_left_x + zone_width)].shape[0]
count_back = count_back_pos + count_back_neg

plt.tight_layout() # Adjust layout to prevent labels from being cut off
plt.show()

# % de permanência durante o jogo
total_points = count_net + count_middle + count_back

if total_points > 0:
    permanencia_rede_pct = (count_net / total_points) * 100
    permanencia_meio_pct = (count_middle / total_points) * 100
    permanencia_fundo_pct = (count_back / total_points) * 100
else:
    permanencia_rede_pct = 0
    permanencia_meio_pct = 0
    permanencia_fundo_pct = 0

print(f"Pontos na Rede: {count_net}")
print(f"Pontos no Meio: {count_middle}")
print(f"Pontos no Fundo: {count_back}")

print(f"\nPorcentagem de permanência na Rede: {permanencia_rede_pct:.2f}%")
print(f"Porcentagem de permanência no Meio: {permanencia_meio_pct:.2f}%")
print(f"Porcentagem de permanência no Fundo: {permanencia_fundo_pct:.2f}%")

# Para encontrar a maior permanência, podemos usar um dicionário para mapear as zonas às suas porcentagens
permanencias = {
    "Rede": permanencia_rede_pct,
    "Meio": permanencia_meio_pct,
    "Fundo": permanencia_fundo_pct
}

# Encontra a zona com a maior porcentagem de permanência
maior_permanencia_zona = max(permanencias, key=permanencias.get)
maior_permanencia_valor = permanencias[maior_permanencia_zona]

print(f"\nA maior permanência em quadra foi na zona {maior_permanencia_zona} com {maior_permanencia_valor:.2f}% dos pontos.")

print(f"{permanencia_rede_pct:.1f}%")
print(f"{permanencia_meio_pct:.1f}%")
print(f"{permanencia_fundo_pct:.1f}%")

# OUTROS GRÁFICOS

#Se a correlação for muito alta, seu corpo está respondendo linearmente ao esforço.

#Se for fraca, pode indicar fadiga ou aquecimento incompleto.

import seaborn as sns

sns.scatterplot(
    data=df, x="Frequência Cardíaca (Bpm)", y="Energia ativa (Kj/min)"
)
plt.title("Correlação FC × Energia ativa")
plt.show()


#FC vs Distância percorrida, Mostra se o aumento de frequência está ligado
#a movimentação ou mais esforço local (braço/saque).
sns.scatterplot(
    data=df, x="Frequência Cardíaca (Bpm)", y="Distância percorrida (Km/min)"
)
plt.title("Correlação FC × Distância percorrida")
plt.show()

#Heatmap de correlações gerais
# Insight:
#Vê rapidamente se FC está mais ligada à energia ou à distância.
#Ideal pra relatórios de TCC (visual e técnico).

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="Reds")
plt.title("Mapa de Correlação entre Variáveis")
plt.show()

#Zonas de intensidade (FC × Energia)
# Insight:
#Você pode ver suas “zonas metabólicas” durante o jogo.
#Excelente para cruzar com resultado (vitória/derrota).

sns.kdeplot(
    data=df,
    x="Frequência Cardíaca (Bpm)",
    y="Energia ativa (Kj/min)",
    fill=True, cmap="Reds"
)
plt.title("Zonas de intensidade - FC x Energia")
plt.show()

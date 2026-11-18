import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Leer los CSV de resultados
df_rag = pd.read_csv("resultados_ragas_coodibot.csv")
df_zero = pd.read_csv("resultados_ragas_zeroshot.csv")

metricas = ["Faithfulness", "Answer Relevancy",
            "Context Precision", "Context Recall"]

zero_means = [df_zero[m].mean() for m in metricas]
rag_means = [df_rag[m].mean() for m in metricas]

# Configurar el gráfico de barras
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(metricas))
width = 0.36

rects1 = ax.bar(x - width/2, zero_means, width,
                label='Línea Base Zero-Shot (gpt-4o-mini sin RAG)', color='#94a3b8', edgecolor='#475569')
rects2 = ax.bar(x + width/2, rag_means, width,
                label='COODIBOT (Advanced RAG)', color='#00bfa5', edgecolor='#00796b')

ax.set_ylabel('Puntaje Promedio (Escala 0.0 – 1.0)',
              fontsize=11, fontweight='bold')
ax.set_title('Comparación de Rendimiento Experimental con Framework RAGAS (N = 50)',
             fontsize=12, fontweight='bold', pad=14)
ax.set_xticks(x)
ax.set_xticklabels(metricas, fontsize=10, fontweight='bold')
ax.set_ylim(0, 0.95)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(frameon=True, fontsize=10, loc='upper left')

# Etiquetas numéricas sobre cada barra
for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#334155', fontweight='bold')

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', fontsize=9, color='#00695c', fontweight='bold')

plt.tight_layout()
plt.savefig("grafico_comparativo_ragas.png", dpi=300)
print("Listo: grafico_comparativo_ragas.png guardado en tu carpeta.")

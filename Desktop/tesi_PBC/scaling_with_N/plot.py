import matplotlib.pyplot as plt
import pandas as pd

# Caricamento e pulizia dei dati dal file CSV
df = pd.read_csv('scaling_con_N - Foglio1.csv')
for col in ['<t> N pari', '<t> N dispari']:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

even_data = df[['N', '<t> N pari']].dropna().sort_values('N')
odd_data = df[['N', '<t> N dispari']].dropna().sort_values('N')

# Creazione della figura
plt.figure(figsize=(6, 4))

# Linee e punti con colori colorblind-proof (Blu #0072B2 e Vermiglio #D55E00)
plt.plot(
    even_data['N'],
    even_data['<t> N pari'],
    color='#0072B2',
    marker='^',
    linestyle='--',
    linewidth=1.5,
    label='even N',
)

plt.plot(
    odd_data['N'],
    odd_data['<t> N dispari'],
    color='#D55E00',
    marker='s',
    linestyle='-',
    linewidth=1.5,
    label='odd N',
)

# Assi, griglia e legenda (nessun titolo inserito)
plt.xlabel('$N$')
plt.ylabel(r'$\langle t_f \rangle$')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('scaling_plot.pdf')
plt.show()
import pandas as pd

# Criando um DataFrame com alguns valores
df = pd.DataFrame({
    'col1': [1, 2, 3],
    'col2': [4, 5, 6]
})

# Somando os elementos de cada coluna
sum_col1 = df['col1'].sum()
sum_col2 = df['col2'].sum()

# Imprimindo os resultados
print('Soma da coluna 1:', sum_col1)
print('Soma da coluna 2:', sum_col2)

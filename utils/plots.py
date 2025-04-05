import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns

COLOR_MAP = {0: 'green', 1: 'yellow', 2: 'orange', 3: 'red', 4: 'darkred'}

def reachable3d(df, rg, max_errors=3):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    df_filtered = df[df['out_of_range'] <= max_errors]

    colors = df_filtered['out_of_range'].map(COLOR_MAP)

    ax.scatter(df_filtered['x'], df_filtered['y'], df_filtered['z'], c=colors, marker='o', s=5)

    ax.set_xlim(-rg, rg)
    ax.set_ylim(-rg, rg)
    ax.set_zlim(-rg, rg)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('Espaço tridimensional alcançável pelo braço robótico')

    plt.show()

def reachable3d_plotly(df, rg, max_errors=3):
    df_filtered = df[df['out_of_range'] <= max_errors]
    
    colors = df_filtered['out_of_range'].map(COLOR_MAP)

    trace = go.Scatter3d(
        x=df_filtered['x'],
        y=df_filtered['y'],
        z=df_filtered['z'],
        mode='markers',
        marker=dict(
            size=5,
            color=colors, 
            opacity=0.8
        )
    )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X', range=[-rg, rg]),
            yaxis=dict(title='Y', range=[-rg, rg]),
            zaxis=dict(title='Z', range=[-rg, rg]),
        ),
        title="Espaço tridimensional alcançável pelo braço robótico",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Gerar o gráfico interativo
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def reachable2d(df, z, rg_x=(-0.25, 0.25), rg_y = (-0.25, 0.25), max_errors=3):
    
    df_clean = df.dropna(subset=["t0", "t1", "t2", "t3"])

    df_filtered = df_clean[df_clean['out_of_range'] <= max_errors]
    
    colors = df_filtered['out_of_range'].map(COLOR_MAP)

    plt.figure(figsize=(8, 8))
    plt.scatter(df_filtered["x"], df_filtered["y"], c=colors, label="Pontos")
    plt.title(f"Distribuição de Pontos em Função de x e y, com z = {z}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(rg_x)
    plt.ylim(rg_y)
    plt.grid(True)
    plt.show()

def check_nulls(x, y, collected_df):
    mask = (collected_df['x'] == x) & (collected_df['y'] == y)
    row = collected_df[mask]
    if row.empty:
        return 'red' 
    else:
        if row.drop(columns=['x', 'y']).isnull().any().any():
            return 'red'
        else:
            return 'green'

def result_df_diff(df_tested, colected_df):
    plt.figure(figsize=(8, 8))

    for _, row in df_tested.iterrows():
        x = row['x']
        y = row['y']

        color = check_nulls(x, y, colected_df)
        plt.scatter(x, y, color=color) 

    plt.xlim(-0.25, 0.25)
    plt.ylim(-0.25, 0.25)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Valores de x, y detectados no teste')
    plt.grid(True)
    plt.show()

def coords_distrib(colected_df):
    plt.figure(figsize=(8, 8))
    plt.scatter(colected_df['xc'], colected_df['yc'], color='blue', s=10)
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.xlabel('xc')
    plt.ylabel('yc')
    plt.title('Distribuição de Coordenadas')
    plt.grid(True)
    plt.show()    
    
def features_distrib(colected_df, columns):
    plt.figure(figsize=(12, 8))

    for i, col in enumerate(columns, 1):
        plt.subplot(2, 3, i)
        sns.histplot(colected_df[col], kde=True, bins=20, color='blue')
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')

    plt.tight_layout()
    plt.show()

def diff_diagonal_normal(df):
    normal = df['diagonal'].mean()
    distances = abs(df['diagonal'] - normal)
    cmap = plt.get_cmap('RdYlGn_r')

    plt.figure(figsize=(10, 6))
    sc = plt.scatter(
        df['x'], 
        df['y'], 
        c=distances, 
        cmap=cmap, 
        s=50, 
        edgecolor='k', 
        alpha=0.8 
    )
    
    nan_mask = df['diagonal'].isna() 
    plt.scatter(df['x'][nan_mask], df['y'][nan_mask], color='black', s=50, edgecolor='k', label='Valores ausentes (NaN)')
    
    plt.title('Distribuição de Pontos com Base na Distância da Normal da Diagonal')
    plt.xlabel('x')
    plt.ylabel('y')

    cbar = plt.colorbar(sc)
    cbar.set_label('Distância da Normal (Diagonal)', rotation=270, labelpad=20)
    
    plt.xlim(0, 0.25)
    plt.ylim(-0.25, 0.25)
    plt.legend()
    plt.grid(True)
    plt.show()

def distribution_and_correlation(df, feature1, feature2):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 3, 1) 
    sns.histplot(df[feature1], kde=True, bins=20, color='blue')
    plt.title(f'Distribuição de {feature1}')
    plt.xlabel(feature1)
    plt.ylabel('Frequência')

    plt.subplot(1, 3, 2)
    sns.histplot(df[feature2], kde=True, bins=20, color='orange')
    plt.title(f'Distribuição de {feature2}')
    plt.xlabel(feature2)
    plt.ylabel('Frequência')

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x=feature1, y=feature2, color='green')
    plt.title(f'Correlação entre {feature1} e {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)

    plt.tight_layout()
    plt.show()

## Pré-requisitos

Certifique-se de ter o Python instalado na sua máquina. Você pode baixá-lo em [python.org](https://www.python.org/).

## Configuração do Ambiente Virtual

1. Crie o ambiente virtual:
   ```sh
   python -m venv .auto
   ```

2. Ative o ambiente virtual:

   - **No Windows**:
     ```sh
     .auto\Scripts\activate
     ```

   - **No macOS/Linux**:
     ```sh
     source .auto/bin/activate
     ```

3. Instale as dependências do projeto:
   ```sh
    pip install -r requirements.txt
   ```

## Executando o Projeto

Depois de configurar o ambiente virtual e instalar as dependências, você pode executar o projeto usando o comando apropriado:

```sh
streamlit run app.py
```

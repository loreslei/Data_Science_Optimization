import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import mean_squared_log_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ====================================================================
# 1. FUNÇÕES DE PRÉ-PROCESSAMENTO (Para uso no Notebook e no Pipeline)
# ====================================================================

def limpar_dados(df, cols_to_drop):
    """
    Remove colunas indesejadas identificadas na Análise Exploratória.
    Essa função será importada no notebook para garantir consistência.
    """
    
    # Mantém apenas as colunas que realmente existem no DataFrame atual (evita erros no teste)
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    return df.drop(columns=cols_to_drop)

def criar_pre_processador(num_features, cat_features):
    """
    Constrói e retorna o ColumnTransformer com as regras de imputação,
    escalonamento e encoding para evitar Data Leakage.
    """
    
    colunas_de_ausencia_conhecidas = [
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'
    ]
    
    # 2. O pipeline divide a lista que veio do notebook automaticamente
    cat_ausencia = [col for col in cat_features if col in colunas_de_ausencia_conhecidas]
    cat_erro = [col for col in cat_features if col not in colunas_de_ausencia_conhecidas]
  
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    ausencia_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    erro_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat_ausencia', ausencia_transformer, cat_ausencia),
        ('cat_erro', erro_transformer, cat_erro)
    ])
    
    return preprocessor

def prever_precos(caminho_arquivo_teste):
    """
    Função obrigatória para o corretor automático.
    Lê o arquivo de teste, aplica o pré-processamento numérico e retorna as predições.
    
    Parâmetros:
    caminho_arquivo_teste (str): Caminho local para o arquivo CSV de teste.
    
    Retorna:
    np.array: As predições de preços (não negativas).
    """
    # 1. Leitura dos dados de teste
    df_teste = pd.read_csv(caminho_arquivo_teste)

    # 2. Pré-processamento (Espelhando o treinamento do baseline)
    # Selecionar apenas colunas numéricas
    X = df_teste.select_dtypes(include=[np.number])
    
    
    cols_to_drop_2 = ['MasVnrArea', 'BsmtFinSF2', 'BsmtUnfSF', 'LowQualFinSF', 'BsmtHalfBath', 'HalfBath', 'BedroomAbvGr','KitchenAbvGr','GarageYrBlt','GarageArea','OpenPorchSF','MiscVal','MoSold','YrSold']
    cols_to_drop = ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    
    df_teste = limpar_dados(df_teste, cols_to_drop)
    df_teste = limpar_dados(df_teste, cols_to_drop_2)
    
    # Remover coluna Id se ela estiver presente
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
        
    # # Preencher valores nulos com 0
    # X = X.fillna(0)

    # 3. Carregamento do modelo
    caminho_modelo = 'modelo_baseline.joblib'
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"O arquivo do modelo '{caminho_modelo}' não foi encontrado na raiz do projeto.")
        
    modelo = joblib.load(caminho_modelo)

    # 4. Alinhamento de colunas (Garante consistência com o treino)
    # if hasattr(modelo, 'feature_names_in_'):
    #     X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)

    # 5. Predição
    predicoes = modelo.predict(X)

    # 6. Pós-processamento
    # Garante valores >= 0 para evitar erro no cálculo do RMSLE
    predicoes_finais = np.clip(predicoes, a_min=0, a_max=None)
    
    
    ### Sugestão
    # # 4. Predição
    # # O pipeline do Scikit-Learn aplica automaticamente a imputação, o scaler e o encoding
    # predicoes_em_log = modelo_pipeline.predict(df_teste)

    # # 5. Pós-processamento
    # # O modelo foi treinado com o alvo em log (np.log1p). 
    # # Precisamos converter de volta para Dólares usando a função exponencial.
    # predicoes_em_dolar = np.expm1(predicoes_em_log)
    
    # # Garante valores >= 0 para evitar qualquer risco na métrica RMSLE
    # predicoes_finais = np.clip(predicoes_em_dolar, a_min=0, a_max=None)

    return predicoes_finais

if __name__ == "__main__":
    # Bloco de teste local para o aluno
    arquivo_teste_exemplo = './datasets/teste_publico.csv'
    
    print(f"--- Executando Validação Local do Pipeline ---")
    
    if not os.path.exists(arquivo_teste_exemplo):
        print(f"[Aviso] Arquivo '{arquivo_teste_exemplo}' não encontrado.")
        print(f"Dica: Crie um CSV fictício com as colunas numéricas para testar o script.")
    else:
        try:
            # Executa a predição
            resultados = prever_precos(arquivo_teste_exemplo)
            
            print("\n✅ Sucesso! O pipeline rodou corretamente.")
            print("-" * 30)
            print("Primeiras 5 predições:")
            print(resultados[:5])
            print("-" * 30)
            
            # Tenta calcular o RMSLE se a coluna alvo estiver no arquivo de teste
            df_val = pd.read_csv(arquivo_teste_exemplo)
            if 'SalePrice' in df_val.columns:
                y_true = df_val['SalePrice']
                # Cálculo do Root Mean Squared Log Error
                rmsle = np.sqrt(mean_squared_log_error(y_true, resultados))
                print(f"Métrica RMSLE Local: {rmsle:.5f}")
            else:
                print("[Nota] Coluna 'SalePrice' não encontrada no CSV. Cálculo do RMSLE pulado.")
            
        except Exception as e:
            print(f"\n❌ Erro encontrado no pipeline:")
            print(str(e))
            print("\nVerifique se o seu modelo espera as mesmas colunas presentes no CSV de teste.")

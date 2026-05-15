import pandas as pd
import joblib
import numpy as np
import os
from sklearn.metrics import mean_squared_log_error

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
    
    df_teste['Idade_Casa'] = df_teste['YrSold'] - df_teste['YearBuilt']
    
    
    COLUNAS_PARA_EXCLUIR = ['Id']

    # Preenchimento de categóricas onde o valor "NA" não é erro, mas sim a ausência do item (ex: Sem Garagem)
    PREENCHIMENTO_CATEGORICAS_CONHECIDAS = {
        'Alley': 'No alley access', 'BsmtQual': 'No Basement', 'BsmtCond': 'No Basement',
        'BsmtExposure': 'No Basement', 'BsmtFinType1': 'No Basement', 'BsmtFinType2': 'No Basement',
        'FireplaceQu': 'No Fireplace', 'GarageType': 'No Garage', 'GarageFinish': 'No Garage',
        'GarageQual': 'No Garage', 'GarageCond': 'No Garage', 'PoolQC': 'No Pool',
        'Fence': 'No Fence', 'MiscFeature': 'None', 'MasVnrType':'None'
    }
    
    
    df_limpo = df_teste.copy()

    # Remove colunas indesejadas
    colunas_excluir_existentes = [col for col in COLUNAS_PARA_EXCLUIR if col in df_limpo.columns]
    df_limpo = df_limpo.drop(columns=colunas_excluir_existentes)

    # Aplica o preenchimento seguro
    for col, valor_preenchimento in PREENCHIMENTO_CATEGORICAS_CONHECIDAS.items():
        if col in df_limpo.columns:
            df_limpo[col] = df_limpo[col].fillna(valor_preenchimento)

    
    # condicao_outlier_area = df_limpo['GrLivArea'] > 4000
    # df_limpo = df_limpo.drop(df_limpo[condicao_outlier_area].index)
    
    # COLUNAS_PARA_APLICAR_IQR = ['LotArea', 'TotalBsmtSF']
   
    # for col in COLUNAS_PARA_APLICAR_IQR:
    #     Q1 = df_limpo[col].quantile(0.25)
    #     Q3 = df_limpo[col].quantile(0.75)
    #     IQR = Q3 - Q1
        
    #     limite_inferior = Q1 - 1.5 * IQR
    #     limite_superior = Q3 + 1.5 * IQR
        
    #     # Mantém apenas quem está dentro dos limites ou é nulo (para ser tratado depois)
    #     df_limpo = df_limpo[
    #         (df_limpo[col] >= limite_inferior) & (df_limpo[col] <= limite_superior) | 
    #         df_limpo[col].isnull()
    #     ]
        
    ESTRATEGIAS_NUMERICAS_POR_COLUNA = {
    'LotFrontage': 'mediana', 
    'MasVnrArea': 'zero',     
    'GarageYrBlt': 'zero'     
    }

    ESTRATEGIA_NUMERICA_PADRAO = 'mediana'
    ESTRATEGIA_CATEGORICA_PADRAO = 'moda'
    
    todas_numericas = df_limpo.select_dtypes(include=[np.number]).columns.tolist()
    todas_categoricas = df_limpo.select_dtypes(exclude=[np.number]).columns.tolist()

    # Aplicar nas colunas NUMÉRICAS
    for col in todas_numericas:
        if df_limpo[col].isnull().sum() > 0:
            estr = ESTRATEGIAS_NUMERICAS_POR_COLUNA.get(col, ESTRATEGIA_NUMERICA_PADRAO)
            if estr == 'zero':
                df_limpo[col] = df_limpo[col].fillna(0)
            elif estr == 'media':
                df_limpo[col] = df_limpo[col].fillna(df_limpo[col].mean())
            elif estr == 'mediana':
                df_limpo[col] = df_limpo[col].fillna(df_limpo[col].median())

    # Aplicar nas colunas CATEGÓRICAS restantes
    for col in todas_categoricas:
        if df_limpo[col].isnull().sum() > 0:
            if ESTRATEGIA_CATEGORICA_PADRAO == 'moda':
                df_limpo[col] = df_limpo[col].fillna(df_limpo[col].mode()[0])
            elif ESTRATEGIA_CATEGORICA_PADRAO == 'ausente':
                df_limpo[col] = df_limpo[col].fillna('Ausente')
    
    target = 'SalePrice'
    cols_num = [c for c in todas_numericas if c != target]
    cols_cat = todas_categoricas

    # --- VERSÃO 1: One-Hot Encoding ---
    df_ohe = pd.get_dummies(df_limpo, columns=cols_cat, drop_first=True)

    # 2. Pré-processamento (Espelhando o treinamento do baseline)
    # Selecionar apenas colunas numéricas
    # X = df_teste.select_dtypes(include=[np.number])
    
    # # Remover coluna Id se ela estiver presente
    # if 'Id' in X.columns:
    #     X = X.drop(columns=['Id'])
        
    # # Preencher valores nulos com 0
    # X = X.fillna(0) 
    
    X_final = df_ohe.copy()
    if 'SalePrice' in X_final.columns:
        X_final = df_ohe.drop(columns=['SalePrice'])

    # 3. Carregamento do modelo
    caminho_modelo = 'modelo_lasso_ohe.joblib'
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"O arquivo do modelo '{caminho_modelo}' não foi encontrado na raiz do projeto.")
        
    modelo = joblib.load(caminho_modelo)
    
    caminho_scaler = 'scaler_lasso_ohe.joblib'

    if not os.path.exists(caminho_scaler):
        raise FileNotFoundError(
            f"O arquivo do scaler '{caminho_scaler}' não foi encontrado."
        )

    scaler = joblib.load(caminho_scaler)

    # # 4. Alinhamento de colunas (Garante consistência com o treino)
    # if hasattr(modelo, 'feature_names_in_'):
    #     X_final = X_final.reindex(columns=modelo.feature_names_in_, fill_value=0)

    # 4. Alinhamento de colunas (Garante consistência com o treino)
    if hasattr(scaler, 'feature_names_in_'):
        X_final = X_final.reindex(columns=scaler.feature_names_in_, fill_value=0)

    
    X_final = scaler.transform(X_final)
    print(X_final.shape)
    # 5. Predição
    predicoes = modelo.predict(X_final)

    # 6. Pós-processamento
    # Garante valores >= 0 para evitar erro no cálculo do RMSLE
    predicoes_finais = np.clip(predicoes, a_min=0, a_max=None)

    return predicoes_finais

if __name__ == "__main__":
    # Bloco de teste local para o aluno
    # arquivo_teste_exemplo = './datasets/treino.csv'
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
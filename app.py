import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from reportlab.lib import colors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ------------------- Funções do agente melhoradas -------------------

def analisar_df(df):
    """Análise completa do dataframe com memória"""
    memoria = {}
    memoria["num_cols"] = df.select_dtypes(include="number").columns.tolist()
    memoria["cat_cols"] = df.select_dtypes(exclude="number").columns.tolist()
    memoria["estatisticas"] = df.describe()
    memoria["frequencias"] = {col: df[col].value_counts() for col in memoria["cat_cols"]}
    memoria["outliers"] = detectar_outliers(df, memoria["num_cols"])
    memoria["df"] = df
    memoria["correlacao"] = df[memoria["num_cols"]].corr() if memoria["num_cols"] else None
    memoria["info_geral"] = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().sum(),
        "duplicadas": df.duplicated().sum()
    }
    memoria["graficos"] = []
    return memoria

def detectar_outliers(df, num_cols):
    """Detecção robusta de outliers"""
    outliers = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = {
            'count': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]),
            'percentual': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]) / len(df) * 100 if len(df) > 0 else 0,
            'valores': df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].tolist()[:10]  # Primeiros 10 valores
        }
    return outliers

def gerar_grafico(df, col, tipo='auto'):
    """Gera gráficos apropriados baseado no tipo de dado"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if tipo == 'auto':
        if pd.api.types.is_numeric_dtype(df[col]):
            # Para colunas numéricas
            if len(df[col].unique()) > 20:
                sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=30)
                ax.set_title(f'Distribuição de {col}')
            else:
                df[col].value_counts().sort_index().head(30).plot.bar(ax=ax)
                ax.set_title(f'Frequência de {col}')
        else:
            # Para colunas categóricas (limita a 30 categorias para não travar)
            df[col].value_counts().head(30).plot.bar(ax=ax)
            ax.set_title(f'Valores mais frequentes - {col}')
            plt.xticks(rotation=45)
    elif tipo == 'boxplot':
        sns.boxplot(y=df[col].dropna(), ax=ax)
        ax.set_title(f'Boxplot - {col}')
    elif tipo == 'scatter':
        if len(df.select_dtypes(include='number').columns) >= 2:
            num_cols = df.select_dtypes(include='number').columns
            ax.scatter(df[num_cols[0]], df[num_cols[1]], alpha=0.6)
            ax.set_xlabel(num_cols[0])
            ax.set_ylabel(num_cols[1])
            ax.set_title(f'Dispersão: {num_cols[0]} vs {num_cols[1]}')
    
    plt.tight_layout()
    return fig

def gerar_heatmap_corr(memoria):
    """Gera heatmap de correlação"""
    if memoria["correlacao"] is not None and len(memoria["num_cols"]) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(memoria["correlacao"], dtype=bool))
        sns.heatmap(memoria["correlacao"], annot=True, fmt=".2f", cmap="coolwarm", 
                   ax=ax, mask=mask, center=0)
        plt.title("Mapa de Correlação entre Variáveis Numéricas")
        plt.tight_layout()
        return fig
    return None

def analisar_tendencias_temporais(df, memoria):
    """Analisa tendências temporais se houver coluna de tempo"""
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'data' in col.lower() or 'timestamp' in col.lower()]
    
    if time_cols:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        time_col = time_cols[0]
        
        # Garantir que coluna de tempo está em formato datetime para plot
        try:
            df_time = df.copy()
            df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
        except Exception:
            df_time = df.copy()
        
        # Tendência temporal
        if len(memoria["num_cols"]) > 0:
            num_col = memoria["num_cols"][0]
            axes[0].plot(df_time[time_col], df_time[num_col], alpha=0.7)
            axes[0].set_xlabel(time_col)
            axes[0].set_ylabel(num_col)
            axes[0].set_title(f'Tendência Temporal: {num_col} vs {time_col}')
        
        # Distribuição temporal (contagens por período)
        axes[1].hist(df_time[time_col].dropna(), bins=50, alpha=0.7)
        axes[1].set_xlabel(time_col)
        axes[1].set_ylabel('Frequência')
        axes[1].set_title(f'Distribuição de {time_col}')
        
        plt.tight_layout()
        return fig, f"Análise temporal realizada usando a coluna '{time_col}'"
    
    return None, "Não foi identificada coluna temporal para análise de tendências"

def identificar_clusters(df, memoria):
    """Identifica clusters nos dados usando KMeans"""
    if len(memoria["num_cols"]) >= 2:
        # Usa as duas primeiras colunas numéricas para clusterização
        X = df[memoria["num_cols"][:2]].dropna()
        
        if len(X) > 0:
            # Padroniza os dados
            scaler = StandardScaler()
            try:
                X_scaled = scaler.fit_transform(X)
            except Exception:
                return None, "Erro ao padronizar dados para clusterização."
            
            # Define número de clusters robusto (n_clusters <= n_amostras)
            n_samples = X.shape[0]
            n_clusters = min(3, n_samples)
            if n_clusters < 1:
                return None, "Amostras insuficientes para clusterização."
            
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
            except Exception as e:
                return None, f"Erro ao executar KMeans: {str(e)}"
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis', alpha=0.6)
            ax.set_xlabel(memoria["num_cols"][0])
            ax.set_ylabel(memoria["num_cols"][1])
            ax.set_title('Identificação de Clusters nos Dados')
            plt.colorbar(scatter)
            plt.tight_layout()
            
            return fig, f"Identificados {len(np.unique(clusters))} clusters nos dados"
    
    return None, "Não há colunas numéricas suficientes para análise de clusters"

def conclusoes_automaticas(memoria):
    """Gera conclusões automáticas baseadas na análise dos dados"""
    df = memoria["df"]
    conclusoes = []
    
    # Informações gerais
    conclusoes.append("📊 **INFORMAÇÕES GERAIS:**")
    conclusoes.append(f"- Dataset com {memoria['info_geral']['shape'][0]} linhas e {memoria['info_geral']['shape'][1]} colunas")
    conclusoes.append(f"- {len(memoria['num_cols'])} variáveis numéricas, {len(memoria['cat_cols'])} variáveis categóricas")
    conclusoes.append(f"- {memoria['info_geral']['missing_values']} valores missing, {memoria['info_geral']['duplicadas']} linhas duplicadas")
    
    # Análise de balanceamento (especialmente para coluna 'Class')
    class_cols = [c for c in df.columns if 'class' in c.lower()]
    if class_cols:
        class_col = class_cols[0]
        counts = df[class_col].value_counts()
        ratio = counts.min() / counts.max() if counts.max() != 0 else 0
        conclusoes.append(f"\n⚖️ **ANÁLISE DE BALANCEAMENTO ({class_col}):**")
        conclusoes.append(f"- Distribuição: {counts.to_dict()}")
        if ratio < 0.1:
            conclusoes.append(f"- ⚠️ ALERTA: Dataset muito desbalanceado (ratio: {ratio:.3f})")
        else:
            conclusoes.append(f"- Dataset relativamente balanceado (ratio: {ratio:.3f})")
    
    # Análise de outliers
    outliers_significativos = {k: v for k, v in memoria["outliers"].items() if v['percentual'] > 5}
    conclusoes.append(f"\n📈 **ANÁLISE DE OUTLIERS:**")
    if outliers_significativos:
        for col, info in list(outliers_significativos.items())[:3]:  # Mostra até 3 colunas
            conclusoes.append(f"- {col}: {info['count']} outliers ({info['percentual']:.1f}%)")
        conclusoes.append("- ⚠️ Recomendação: Investigar impacto dos outliers na análise")
    else:
        conclusoes.append("- ✅ Poucos outliers detectados")
    
    # Análise de correlação
    if memoria["correlacao"] is not None:
        conclusoes.append(f"\n🔗 **ANÁLISE DE CORRELAÇÃO:**")
        # Encontra correlações fortes (abs > 0.7)
        strong_corr = []
        for i in range(len(memoria["correlacao"].columns)):
            for j in range(i+1, len(memoria["correlacao"].columns)):
                corr_val = memoria["correlacao"].iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corr.append((memoria["correlacao"].columns[i], 
                                      memoria["correlacao"].columns[j], corr_val))
        
        if strong_corr:
            for var1, var2, corr in strong_corr[:3]:  # Mostra até 3 correlações fortes
                conclusoes.append(f"- {var1} ↔ {var2}: {corr:.3f}")
        else:
            conclusoes.append("- Nenhuma correlação forte detectada (|r| > 0.7)")
    
    # Recomendações
    conclusoes.append(f"\n💡 **RECOMENDAÇÕES:**")
    if memoria['info_geral']['missing_values'] > 0:
        conclusoes.append("- Tratar valores missing antes de modelagem")
    if outliers_significativos:
        conclusoes.append("- Avaliar impacto dos outliers nas análises")
    if class_cols and ratio < 0.1:
        conclusoes.append("- Considerar técnicas para dados desbalanceados (oversampling, undersampling)")
    
    return "\n".join(conclusoes)

# ------------------- Funções para PDF -------------------

def adicionar_grafico_pdf(c, fig, y, width=400, height=300):
    """Adiciona gráfico ao PDF"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = ImageReader(buf)
    c.drawImage(img, 50, y - height - 20, width=width, height=height)
    buf.close()
    return y - height - 40

def gerar_pdf(memoria, perguntas_respostas):
    """Gera relatório PDF completo"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Configurações iniciais
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Relatório de Análise de Dados - Agente I2A2")
    y -= 30
    
    # Informações gerais
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "📊 Informações Gerais do Dataset:")
    y -= 20
    c.setFont("Helvetica", 10)
    
    info_lines = [
        f"Total de linhas: {memoria['info_geral']['shape'][0]}",
        f"Total de colunas: {memoria['info_geral']['shape'][1]}",
        f"Colunas numéricas: {len(memoria['num_cols'])}",
        f"Colunas categóricas: {len(memoria['cat_cols'])}",
        f"Valores missing: {memoria['info_geral']['missing_values']}",
        f"Linhas duplicadas: {memoria['info_geral']['duplicadas']}"
    ]
    
    for line in info_lines:
        if y < 100:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
        c.drawString(70, y, line)
        y -= 15
    
    y -= 10
    
    # Perguntas e respostas
    c.setFont("Helvetica-Bold", 12)
    if y < 100:
        c.showPage()
        y = height - 50
    c.drawString(50, y, "❓ Análises Realizadas:")
    y -= 25
    
    for i, (pergunta, resposta, fig) in enumerate(perguntas_respostas, 1):
        # Pergunta
        c.setFont("Helvetica-Bold", 10)
        if y < 100:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, f"Pergunta {i}: {pergunta}")
        y -= 15
        
        # Resposta
        c.setFont("Helvetica", 9)
        lines = resposta.split('\n')
        for line in lines:
            if line.strip():  # Ignora linhas vazias
                # Limpa emojis e formatação para o PDF
                clean_line = line.replace('**', '').replace('📊', '').replace('📈', '').replace('🔍', '')
                clean_line = clean_line.replace('👥', '').replace('🔗', '').replace('⚖️', '').replace('💡', '')
                clean_line = clean_line.replace('✅', '').replace('⚠️', '').replace('🤔', '').strip()
                
                if clean_line:
                    if y < 50:
                        c.showPage()
                        y = height - 50
                        c.setFont("Helvetica", 9)
                    
                    # Quebra linha se muito longa
                    if len(clean_line) > 100:
                        words = clean_line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line + " " + word) <= 100:
                                current_line += " " + word
                            else:
                                c.drawString(70, y, current_line.strip())
                                y -= 12
                                current_line = word
                        if current_line:
                            c.drawString(70, y, current_line.strip())
                            y -= 12
                    else:
                        c.drawString(70, y, clean_line)
                        y -= 12
        
        y -= 5
        
        # Gráfico
        if fig is not None:
            if y < 300:  # Se não há espaço suficiente para o gráfico
                c.showPage()
                y = height - 50
            y = adicionar_grafico_pdf(c, fig, y)
        
        y -= 15
    
    # Adiciona conclusões automáticas se não estiverem presentes
    tem_conclusoes = any('conclusão' in p[0].lower() for p in perguntas_respostas)
    if not tem_conclusoes:
        if y < 150:
            c.showPage()
            y = height - 50
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "🤔 Conclusões Automáticas:")
        y -= 20
        
        conclusoes = conclusoes_automaticas(memoria)
        c.setFont("Helvetica", 9)
        lines = conclusoes.split('\n')
        for line in lines:
            if line.strip():
                clean_line = line.replace('**', '').replace('📊', '').replace('⚖️', '').replace('📈', '')
                clean_line = clean_line.replace('🔗', '').replace('💡', '').replace('✅', '').replace('⚠️', '').strip()
                
                if clean_line:
                    if y < 50:
                        c.showPage()
                        y = height - 50
                        c.setFont("Helvetica", 9)
                    
                    if len(clean_line) > 100:
                        words = clean_line.split()
                        current_line = ""
                        for word in words:
                            if len(current_line + " " + word) <= 100:
                                current_line += " " + word
                            else:
                                c.drawString(70, y, current_line.strip())
                                y -= 12
                                current_line = word
                        if current_line:
                            c.drawString(70, y, current_line.strip())
                            y -= 12
                    else:
                        c.drawString(70, y, clean_line)
                        y -= 12
    
    c.save()
    buffer.seek(0)
    return buffer

# ------------------- Interface Streamlit Melhorada -------------------

st.set_page_config(page_title="Agente de Análise de CSV - I2A2", layout="wide")

st.title("🤖 Agente de Análise de Arquivos CSV - I2A2")
st.markdown("### 📊 Trabalho de Inteligência Artificial Aplicada")

# Sidebar com informações
with st.sidebar:
    st.header("ℹ️ Sobre o Agente")
    st.markdown("""
    Este agente foi desenvolvido para a **Atividade Obrigatória** do I2A2.
    
    **Capacidades:**
    - Análise exploratória completa
    - Detecção de outliers
    - Identificação de clusters
    - Análise de correlação
    - Tendências temporais
    - Conclusões automáticas
    
    **Funcionalidades:**
    - Upload de qualquer CSV
    - Respostas a perguntas naturais
    - Geração de gráficos
    - Relatório PDF completo
    """)
    
    st.header("🎯 Perguntas Sugeridas")
    if st.button("Tipos de dados e estrutura"):
        st.session_state.pergunta = "Quais são os tipos de dados e estrutura do dataset?"
    if st.button("Distribuições e estatísticas"):
        st.session_state.pergunta = "Mostre as distribuições e estatísticas descritivas"
    if st.button("Análise de outliers"):
        st.session_state.pergunta = "Existem outliers nos dados? Como afetam a análise?"
    if st.button("Correlações entre variáveis"):
        st.session_state.pergunta = "Quais as correlações entre as variáveis?"
    if st.button("Identificar clusters"):
        st.session_state.pergunta = "Existem agrupamentos nos dados?"
    if st.button("Conclusões automáticas"):
        st.session_state.pergunta = "Quais são as conclusões do agente sobre os dados?"

# Área principal
uploaded_file = st.file_uploader("📁 Carregue um arquivo CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Dataset carregado com sucesso: {df.shape[0]} linhas × {df.shape[1]} colunas")
        
        # Se for um novo arquivo (ou primeira vez), reanalisar e resetar histórico
        if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
            st.session_state.memoria = analisar_df(df)
            st.session_state.perguntas_respostas = []
            st.session_state.last_uploaded_file = uploaded_file.name
        else:
            # Em caso de recarregar o mesmo arquivo, atualiza a memória também (mantém histórico)
            st.session_state.memoria = analisar_df(df)
            if 'perguntas_respostas' not in st.session_state:
                st.session_state.perguntas_respostas = []
        
        # Abas para organização
        tab1, tab2, tab3 = st.tabs(["📋 Visualização", "❓ Análise Interativa", "📊 Análise Rápida"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Pré-visualização dos Dados")
                st.dataframe(df.head(10))
            with col2:
                st.subheader("Informações Gerais")
                st.write(f"**Formato:** {df.shape[0]} linhas × {df.shape[1]} colunas")
                st.write(f"**Colunas numéricas:** {len(df.select_dtypes(include='number').columns)}")
                st.write(f"**Colunas categóricas:** {len(df.select_dtypes(exclude='number').columns)}")
                st.write(f"**Valores missing:** {df.isnull().sum().sum()}")
                st.write(f"**Linhas duplicadas:** {df.duplicated().sum()}")
        
        memoria = st.session_state.memoria
        
        with tab3:
            st.subheader("🔍 Análise Rápida Automática")
            if st.button("Executar Análise Completa"):
                with st.spinner("Analisando dados..."):
                    # Gera análise completa
                    fig_dist, _ = plt.subplots(figsize=(12, 8))
                    if memoria["num_cols"]:
                        # hist cria sua própria figura; mostramos a figura corrente (fig_dist) apenas para compatibilidade
                        df[memoria["num_cols"]].hist(figsize=(12, 8), bins=20)
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                    
                    if memoria["correlacao"] is not None:
                        fig_corr = gerar_heatmap_corr(memoria)
                        if fig_corr:
                            st.pyplot(fig_corr)
                    
                    # Mostra conclusões
                    st.subheader("🤔 Conclusões Automáticas")
                    conclusoes = conclusoes_automaticas(memoria)
                    st.text_area("Conclusões:", conclusoes, height=300)
        
        with tab2:
            st.subheader("💬 Faça sua pergunta")
            
            # Input de pergunta
            pergunta = st.text_input(
                "Digite sua pergunta sobre os dados:",
                value=st.session_state.get('pergunta', ''),
                key="pergunta_input"
            )
            
            if pergunta:
                with st.spinner("Processando sua pergunta..."):
                    resposta = ""
                    fig = None
                    pergunta_lower = pergunta.lower()
                    
                    # --- REGRAS DE PERGUNTAS MELHORADAS ---
                    
                    # PRIMEIRO: Verifica se é uma pergunta sobre estatísticas específicas de uma coluna
                    coluna_encontrada = None
                    estatistica_solicitada = None
                    
                    # Procura por nomes de colunas na pergunta
                    for col in df.columns:
                        if col.lower() in pergunta_lower:
                            coluna_encontrada = col
                            break
                    
                    # Detecta qual estatística está sendo solicitada
                    if coluna_encontrada:
                        if any(palavra in pergunta_lower for palavra in ['média', 'media', 'average', 'mean']):
                            estatistica_solicitada = 'media'
                        elif any(palavra in pergunta_lower for palavra in ['mediana', 'median']):
                            estatistica_solicitada = 'mediana'
                        elif any(palavra in pergunta_lower for palavra in ['desvio', 'padrão', 'padrao', 'std']):
                            estatistica_solicitada = 'desvio_padrao'
                        elif any(palavra in pergunta_lower for palavra in ['variância', 'variancia', 'variability']):
                            estatistica_solicitada = 'variancia'
                        elif any(palavra in pergunta_lower for palavra in ['mínimo', 'minimo', 'min', 'minimum']):
                            estatistica_solicitada = 'minimo'
                        elif any(palavra in pergunta_lower for palavra in ['máximo', 'maximo', 'max', 'maximum']):
                            estatistica_solicitada = 'maximo'
                        elif any(palavra in pergunta_lower for palavra in ['soma', 'sum', 'total']):
                            estatistica_solicitada = 'soma'
                    
                    # Se encontrou coluna e estatística, responde diretamente
                    if coluna_encontrada and estatistica_solicitada:
                        if coluna_encontrada in memoria["num_cols"]:
                            if estatistica_solicitada == 'media':
                                valor = df[coluna_encontrada].mean()
                                resposta = f"📊 A média da coluna **{coluna_encontrada}** é: **{valor:.4f}**"
                            elif estatistica_solicitada == 'mediana':
                                valor = df[coluna_encontrada].median()
                                resposta = f"📊 A mediana da coluna **{coluna_encontrada}** é: **{valor:.4f}**"
                            elif estatistica_solicitada == 'desvio_padrao':
                                valor = df[coluna_encontrada].std()
                                resposta = f"📊 O desvio padrão da coluna **{coluna_encontrada}** é: **{valor:.4f}**"
                            elif estatistica_solicitada == 'variancia':
                                valor = df[coluna_encontrada].var()
                                resposta = f"📊 A variância da coluna **{coluna_encontrada}** é: **{valor:.4f}**"
                            elif estatistica_solicitada == 'minimo':
                                valor = df[coluna_encontrada].min()
                                resposta = f"📊 O valor mínimo da coluna **{coluna_encontrada}** é: **{valor:.4f}**"
                            elif estatistica_solicitada == 'maximo':
                                valor = df[coluna_encontrada].max()
                                resposta = f"📊 O valor máximo da coluna **{coluna_encontrada}** é: **{valor:.4f}**"
                            elif estatistica_solicitada == 'soma':
                                valor = df[coluna_encontrada].sum()
                                resposta = f"📊 A soma total da coluna **{coluna_encontrada}** é: **{valor:.4f}**"
                            
                            # Adiciona informações extras
                            resposta += f"\n\n📈 **Estatísticas completas de {coluna_encontrada}:**"
                            stats = df[coluna_encontrada].describe()
                            resposta += f"\n- Média: {stats['mean']:.4f}"
                            resposta += f"\n- Mediana: {stats['50%']:.4f}"
                            resposta += f"\n- Desvio Padrão: {stats['std']:.4f}"
                            resposta += f"\n- Mínimo: {stats['min']:.4f}"
                            resposta += f"\n- Máximo: {stats['max']:.4f}"
                            resposta += f"\n- Q1 (25%): {stats['25%']:.4f}"
                            resposta += f"\n- Q3 (75%): {stats['75%']:.4f}"
                            
                            # Gera gráfico da coluna
                            fig = gerar_grafico(df, coluna_encontrada)
                            
                        else:
                            # Para colunas categóricas
                            if estatistica_solicitada in ['media', 'mediana', 'desvio_padrao', 'variancia']:
                                resposta = f"⚠️ A coluna **{coluna_encontrada}** é categórica, não é possível calcular {estatistica_solicitada}."
                            else:
                                freq = df[coluna_encontrada].value_counts()
                                resposta = f"📊 **Análise da coluna categórica {coluna_encontrada}:**\n\n"
                                resposta += f"- Total de categorias: {len(freq)}\n"
                                resposta += f"- Valor mais frequente: {freq.index[0]} ({freq.iloc[0]} ocorrências)\n"
                                resposta += f"- Valor menos frequente: {freq.index[-1]} ({freq.iloc[-1]} ocorrências)\n\n"
                                resposta += "**Top 5 valores mais frequentes:**\n"
                                for i, (valor, count) in enumerate(freq.head().items()):
                                    percentual = (count / len(df)) * 100
                                    resposta += f"{i+1}. {valor}: {count} ({percentual:.1f}%)\n"
                                
                                fig = gerar_grafico(df, coluna_encontrada)
                    
                    # SE NÃO ENCONTROU ESTATÍSTICA ESPECÍFICA, continua com as regras originais
                    elif not resposta:
                        # Tipos de dados e estrutura
                        if any(palavra in pergunta_lower for palavra in ['tipo', 'dados', 'estrutura', 'coluna']):
                            resposta = "**📝 ESTRUTURA DO DATASET:**\n\n"
                            resposta += f"- Total de colunas: {len(df.columns)}\n"
                            resposta += f"- Total de linhas: {len(df)}\n\n"
                            
                            resposta += "**Colunas Numéricas:**\n"
                            for col in memoria["num_cols"]:
                                resposta += f"- {col} ({df[col].dtype})\n"
                            
                            resposta += "\n**Colunas Categóricas:**\n"
                            for col in memoria["cat_cols"]:
                                resposta += f"- {col} ({df[col].dtype})\n"
                        
                        # Distribuições e estatísticas
                        elif any(palavra in pergunta_lower for palavra in ['distribuição', 'histograma', 'estatística', 'descritiv']):
                            if memoria["num_cols"]:
                                resposta = "**📊 ESTATÍSTICAS DESCRITIVAS:**\n\n"
                                stats = df[memoria["num_cols"]].describe()
                                for col in stats.columns:
                                    resposta += f"**{col}:**\n"
                                    resposta += f"- Média: {stats[col]['mean']:.2f}\n"
                                    resposta += f"- Mediana: {stats[col]['50%']:.2f}\n"
                                    resposta += f"- Desvio Padrão: {stats[col]['std']:.2f}\n"
                                    resposta += f"- Mínimo: {stats[col]['min']:.2f}\n"
                                    resposta += f"- Máximo: {stats[col]['max']:.2f}\n\n"
                                
                                # Gera histogramas
                                if len(memoria["num_cols"]) <= 6:
                                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                                    axes = axes.flatten()
                                    for i, col in enumerate(memoria["num_cols"][:6]):
                                        if i < len(axes):
                                            df[col].hist(ax=axes[i], bins=20, alpha=0.7)
                                            axes[i].set_title(f'Distribuição de {col}')
                                    plt.tight_layout()
                            else:
                                resposta = "Não há colunas numéricas para análise estatística."
                        
                        # Outliers
                        elif any(palavra in pergunta_lower for palavra in ['outlier', 'atípico', 'anomalia']):
                            resposta = "**🔍 ANÁLISE DE OUTLIERS:**\n\n"
                            outliers_detectados = False
                            
                            for col, info in memoria["outliers"].items():
                                if info['count'] > 0:
                                    outliers_detectados = True
                                    resposta += f"**{col}:**\n"
                                    resposta += f"- {info['count']} outliers ({info['percentual']:.1f}% dos dados)\n"
                                    if info['percentual'] > 5:
                                        resposta += f"- ⚠️ Impacto significativo esperado\n"
                                    resposta += "\n"
                            
                            if not outliers_detectados:
                                resposta += "✅ Poucos ou nenhum outlier detectado nos dados.\n"
                            else:
                                resposta += "\n**Recomendações:**\n"
                                resposta += "- Investigar se outliers representam erros ou casos especiais\n"
                                resposta += "- Avaliar impacto nas análises estatísticas\n"
                                resposta += "- Considerar transformações ou remoção se necessário"
                        
                        # Correlação
                        elif any(palavra in pergunta_lower for palavra in ['correlação', 'relação', 'associação']):
                            fig = gerar_heatmap_corr(memoria)
                            if fig:
                                resposta = "**🔗 MAPA DE CORRELAÇÃO:**\n\n"
                                resposta += "Heatmap gerado mostrando as correlações entre variáveis numéricas.\n"
                                resposta += "Valores próximos de +1 indicam correlação positiva forte.\n"
                                resposta += "Valores próximos de -1 indicam correlação negativa forte.\n"
                                resposta += "Valores próximos de 0 indicam pouca ou nenhuma correlação."
                            else:
                                resposta = "Não há colunas numéricas suficientes para análise de correlação."
                        
                        # Clusters
                        elif any(palavra in pergunta_lower for palavra in ['cluster', 'agrupamento', 'grupo']):
                            fig, cluster_info = identificar_clusters(df, memoria)
                            resposta = f"**👥 ANÁLISE DE CLUSTERS:**\n\n{cluster_info}"
                        
                        # Tendências temporais
                        elif any(palavra in pergunta_lower for palavra in ['tendência', 'temporal', 'time', 'sazonal']):
                            fig, trend_info = analisar_tendencias_temporais(df, memoria)
                            resposta = f"**📈 ANÁLISE TEMPORAL:**\n\n{trend_info}"
                        
                        # Conclusões
                        elif any(palavra in pergunta_lower for palavra in ['conclusão', 'conclusoes', 'resumo', 'insight']):
                            resposta = conclusoes_automaticas(memoria)
                        
                        # Gráfico específico
                        elif any(palavra in pergunta_lower for palavra in ['gráfico', 'plot', 'visualizar', 'mostrar']):
                            coluna_encontrada = None
                            for col in df.columns:
                                if col.lower() in pergunta_lower:
                                    coluna_encontrada = col
                                    break
                            
                            if coluna_encontrada:
                                fig = gerar_grafico(df, coluna_encontrada)
                                resposta = f"**📈 GRÁFICO DE {coluna_encontrada.upper()}:**\n\nGráfico gerado com sucesso."
                            else:
                                resposta = "Por favor, especifique qual coluna você gostaria de visualizar."
                        
                        # Pergunta não reconhecida
                        else:
                            resposta = "🤔 Não entendi completamente sua pergunta. Tente ser mais específico ou usar palavras como:\n\n"
                            resposta += "- 'média da coluna X', 'mediana de Y'\n"
                            resposta += "- 'tipos de dados', 'estrutura'\n"
                            resposta += "- 'distribuição', 'estatísticas'\n" 
                            resposta += "- 'outliers', 'anomalias'\n"
                            resposta += "- 'correlação', 'relação entre variáveis'\n"
                            resposta += "- 'clusters', 'agrupamentos'\n"
                            resposta += "- 'tendências temporais'\n"
                            resposta += "- 'conclusões', 'insights'\n"
                            resposta += "- 'gráfico de [nome da coluna]'"
                    
                    # Exibe resposta
                    # garante que perguntas_respostas exista
                    if 'perguntas_respostas' not in st.session_state:
                        st.session_state.perguntas_respostas = []
                    
                    st.subheader("💡 Resposta do Agente")
                    st.text_area("Resposta:", resposta, height=200, key=f"resposta_{len(st.session_state.perguntas_respostas)}")
                    
                    if fig:
                        st.pyplot(fig)
                    
                    # Armazena para PDF
                    st.session_state.perguntas_respostas.append((pergunta, resposta, fig))
                    
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")

# Geração do PDF
if 'perguntas_respostas' in st.session_state and st.session_state.perguntas_respostas:
    st.markdown("---")
    st.subheader("📄 Relatório PDF")
    
    if st.button("📥 Gerar Relatório PDF Completo"):
        with st.spinner("Gerando relatório PDF..."):
            # Adiciona análise completa ao PDF
            perguntas_completas = st.session_state.perguntas_respostas.copy()
            
            # Adiciona conclusões automáticas se não estiverem presentes
            tem_conclusoes = any('conclusão' in p[0].lower() for p in perguntas_completas)
            if not tem_conclusoes:
                conclusoes = conclusoes_automaticas(st.session_state.memoria)
                perguntas_completas.append(("Conclusões automáticas do agente", conclusoes, None))
            
            pdf_buffer = gerar_pdf(st.session_state.memoria, perguntas_completas)
            
            st.success("✅ Relatório PDF gerado com sucesso!")
            st.download_button(
                label="⬇️ Download do Relatório PDF",
                data=pdf_buffer,
                file_name="Relatorio_Agente_I2A2.pdf",
                mime="application/pdf"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🤖 Agente de Análise de CSV - Desenvolvido para I2A2 - Atividade Obrigatória 2025
    </div>
    """,
    unsafe_allow_html=True
)

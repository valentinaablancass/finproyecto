import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Análisis de Portafolios", layout="wide")
st.title("Análisis y Optimización de Portafolios")

# Definición de ETFs y ventanas de tiempo
etfs = ['LQD', 'EMB', 'SPY', 'EMXC', 'IAU']
descripciones_etfs = {
    "LQD": {
        "nombre": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "exposicion": "Bonos corporativos denominados en dólares de EE.UU. y con grado de inversión.",
        "indice": "iBoxx $ Liquid Investment Grade Index",
        "moneda": "USD",
        "principales": ["JPMorgan Chase & Co", "Bank of America Corp", "Morgan Stanley"],
        "paises": "Estados Unidos y empresas multinacionales",
        "estilo": "Value",
        "costos": "Comisión de administración: 0.14%"
    },
    "EMB": {
        "nombre": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "exposicion": "Bonos de gobierno denominados en USD emitidos por países de mercados emergentes.",
        "indice": "JPMorgan EMBI Global Core Index",
        "moneda": "USD",
        "principales": ["Turkey (Republic of)", "Saudi Arabia (Kingdom of)", "Brazil Federative Republic of"],
        "paises": "América Latina, Medio Oriente, África y Asia",
        "estilo": "Value",
        "costos": "Comisión de administración: 0.39%"
    },
    "SPY": {
        "nombre": "iShares Core S&P 500 ETF",
        "exposicion": "Empresas de alta capitalización en Estados Unidos.",
        "indice": "S&P 500 Index (USD)",
        "moneda": "USD",
        "principales": ["Apple Inc", "NVIDIA Corp", "Microsoft Corp"],
        "paises": "Estados Unidos",
        "estilo": "Mix (Growth/Value)",
        "costos": "Comisión de administración: 0.03%"
    },
    "EMXC": {
        "nombre": "iShares MSCI Emerging Markets ex China ETF",
        "exposicion": "Empresas de mercados emergentes excluyendo China.",
        "indice": "MSCI Emerging Markets ex China Index",
        "moneda": "USD",
        "principales": ["Samsung Electronics", "Taiwan Semiconductor", "Infosys"],
        "paises": "Corea del Sur, Taiwán, India, entre otros",
        "estilo": "Growth",
        "costos": "Comisión de administración: 0.25%"
    },
    "IAU": {
        "nombre": "iShares Gold Trust",
        "exposicion": "Inversión en oro físico como cobertura inflacionaria.",
        "indice": "N/A",
        "moneda": "USD",
        "principales": ["N/A"],
        "paises": "Global",
        "estilo": "Commodity",
        "costos": "Comisión de administración: 0.25%"
    }
}
ventanas = {
    "2010-2023": ("2010-01-01", "2023-12-31"),
    "2010-2020": ("2010-01-01", "2020-12-31"),
    "2021-2023": ("2021-01-01", "2023-12-31")
}

# Selección de ventana de tiempo en el sidebar
st.sidebar.header("Configuración de Ventana")
ventana = st.sidebar.selectbox("Selecciona una ventana de tiempo para análisis:", options=list(ventanas.keys()))
start_date, end_date = ventanas[ventana]

# Descargar datos de los ETFs
@st.cache_data
def obtener_datos(etfs, start_date, end_date):
    datos = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    return datos

# Descargar datos del tipo de cambio USD/MXN
@st.cache_data
def obtener_tipo_cambio(start_date, end_date):
    tipo_cambio = yf.download('USDMXN=X', start=start_date, end=end_date)['Adj Close']
    return tipo_cambio

# Obtener datos
datos = obtener_datos(etfs, start_date, end_date)
tipo_cambio = obtener_tipo_cambio(start_date, end_date)

# Calcular rendimientos diarios
rendimientos = datos.pct_change().dropna()

# Tabs de la Aplicación
tab1, tab2, tab3, tab4 = st.tabs([
    "Análisis de Activos Individuales",
    "Portafolios Óptimos",
    "Comparación de Portafolios",
    "Black - Litterman"
])

# Tab 1: Análisis de Activos Individuales
with tab1:
    st.header("Análisis de Activos Individuales")
    etf_seleccionado = st.selectbox("Selecciona un ETF para análisis:", options=etfs)

    if etf_seleccionado:
        if etf_seleccionado not in datos.columns or datos[etf_seleccionado].dropna().empty:
            st.error(f"No hay datos disponibles para {etf_seleccionado} en la ventana seleccionada.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Características del ETF")
                data = descripciones_etfs[etf_seleccionado]
                st.write("**Nombre:**", data["nombre"])
                st.write("**Exposición:**", data["exposicion"])
                st.write("**Índice:**", data["indice"])
                st.write("**Moneda:**", data["moneda"])
                st.write("**Principales Contribuyentes:**", ", ".join(data["principales"]))
                st.write("**Países:**", data["paises"])
                st.write("**Estilo:**", data["estilo"])
                st.write("**Costos:**", data["costos"])

            with col2:
                st.subheader("Métricas Calculadas")
                def calcular_metricas(rendimientos):
                    media = rendimientos.mean() * 252
                    volatilidad = rendimientos.std() * np.sqrt(252)
                    sharpe = media / volatilidad
                    sesgo = rendimientos.skew()
                    curtosis = rendimientos.kurt()
                    return {
                        "Media": media,
                        "Volatilidad": volatilidad,
                        "Sharpe": sharpe,
                        "Sesgo": sesgo,
                        "Curtosis": curtosis
                    }

                metricas = calcular_metricas(rendimientos[etf_seleccionado])
                for key, value in metricas.items():
                    st.metric(key, f"{value:.2f}" if key != "Sharpe" else f"{value:.2f}")

            st.subheader("Serie de Tiempo de Precios Normalizados")
            precios_normalizados = datos[etf_seleccionado] / datos[etf_seleccionado].iloc[0] * 100
            fig = go.Figure(go.Scatter(
                x=precios_normalizados.index,
                y=precios_normalizados,
                mode='lines',
                name=etf_seleccionado
            ))
            fig.update_layout(
                title="Precio Normalizado",
                xaxis_title="Fecha",
                yaxis_title="Precio Normalizado",
                hovermode="x unified"
            )
            st.plotly_chart(fig)

# Tab 2: Portafolios Óptimos
with tab2:
    st.header("Portafolios Óptimos")

    def optimizar_portafolio(rendimientos, objetivo="sharpe", rendimiento_objetivo=None, incluir_tipo_cambio=False):
        media = rendimientos.mean() * 252
        covarianza = rendimientos.cov() * 252
        num_activos = len(media)
        pesos_iniciales = np.ones(num_activos) / num_activos
        limites = [(0.035, 0.4) for _ in range(num_activos)]
        restricciones = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if incluir_tipo_cambio:
            tipo_cambio_rendimientos = tipo_cambio.pct_change().mean() * 252
            media += tipo_cambio_rendimientos

        if objetivo == "sharpe":
            def objetivo_func(pesos):
                rendimiento = np.dot(pesos, media)
                riesgo = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
                return -rendimiento / riesgo
        elif objetivo == "volatilidad":
            def objetivo_func(pesos):
                return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
        elif objetivo == "rendimiento":
            restricciones.append({'type': 'eq', 'fun': lambda x: np.dot(x, media) - rendimiento_objetivo})
            def objetivo_func(pesos):
                return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))

        resultado = sco.minimize(objetivo_func, pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)
        return resultado.x

    pesos_sharpe = optimizar_portafolio(rendimientos, objetivo="sharpe")
    pesos_volatilidad = optimizar_portafolio(rendimientos, objetivo="volatilidad")
    pesos_rendimiento = optimizar_portafolio(rendimientos, objetivo="rendimiento", rendimiento_objetivo=0.10, incluir_tipo_cambio=True)

    pesos_df = pd.DataFrame({
        "Máximo Sharpe": pesos_sharpe,
        "Mínima Volatilidad": pesos_volatilidad,
        "Mínima Volatilidad (Rendimiento 10% en MXN)": pesos_rendimiento
    }, index=etfs)

    portafolio_seleccionado = st.selectbox(
        "Selecciona el portafolio a visualizar:",
        ["Máximo Sharpe", "Mínima Volatilidad", "Mínima Volatilidad (Rendimiento 10% en MXN)"]
    )

    st.subheader(f"Pesos del Portafolio: {portafolio_seleccionado}")
    st.bar_chart(pesos_df[portafolio_seleccionado])

    st.subheader("Distribución del Portafolio")
    fig = go.Figure(data=[
        go.Pie(labels=etfs, values=pesos_df[portafolio_seleccionado], hoverinfo='label+percent')
    ])
    fig.update_layout(title=f"Composición del Portafolio: {portafolio_seleccionado}")
    st.plotly_chart(fig)

    if portafolio_seleccionado == "Mínima Volatilidad (Rendimiento 10% en MXN)":
        st.subheader("Ajustes por Tipo de Cambio")
        try:
            tipo_cambio_medio = tipo_cambio.mean()
            if isinstance(tipo_cambio_medio, pd.Series):
                tipo_cambio_medio = tipo_cambio_medio.iloc[0]
            st.write(f"**Tipo de cambio medio esperado:** {tipo_cambio_medio:.2f} USD/MXN")
        except Exception as e:
            st.error(f"Error al calcular el promedio del tipo de cambio: {e}")

with tab3:
    st.header("Comparación de Portafolios")

    st.subheader("Precios Normalizados de los ETFs Seleccionados")
    precios_normalizados = datos / datos.iloc[0] * 100

    fig = go.Figure()
    for etf in etfs:
        fig.add_trace(go.Scatter(
            x=precios_normalizados.index,
            y=precios_normalizados[etf],
            mode='lines',
            name=etf
        ))

    fig.update_layout(
        title="Comparación de Precios Normalizados",
        xaxis_title="Fecha",
        yaxis_title="Precio Normalizado",
        hovermode="x unified"
    )
    st.plotly_chart(fig)
with tab4:
    st.header("Modelo de Optimización Black-Litterman")

    try:
        # Definición de matrices y parámetros
        media_rendimientos = rendimientos.mean() * 252
        covarianza_rendimientos = rendimientos.cov() * 252

        # Matriz de views (P) y rendimientos esperados (Q)
        P = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ])
        Q = np.array([0.08, 0.065, 0.12, 0.06, 0.05])  # Rendimientos esperados para cada activo

        def black_litterman_optimizar(media_rendimientos, covarianza_rendimientos, P, Q, tau=0.05):
            """
            Optimización usando el modelo Black-Litterman.
            - media_rendimientos: Rendimientos esperados a priori (pi).
            - covarianza_rendimientos: Matriz de covarianza de los rendimientos.
            - P: Matriz de views (opiniones) sobre los activos.
            - Q: Vector de rendimientos esperados basado en los views.
            - tau: Escalador de incertidumbre de la distribución a priori.
            """
            pi = media_rendimientos  # Rendimientos esperados a priori
            omega = np.diag(np.diag(P @ covarianza_rendimientos @ P.T)) * tau  # Incertidumbre en los views

            # Ajuste de Black-Litterman
            medio_ajustado = np.linalg.inv(
                np.linalg.inv(tau * covarianza_rendimientos) + P.T @ np.linalg.inv(omega) @ P
            ) @ (
                np.linalg.inv(tau * covarianza_rendimientos) @ pi + P.T @ np.linalg.inv(omega) @ Q
            )

            restricciones = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Pesos suman 1

            def objetivo_func(pesos):
                # Maximizar el rendimiento ajustado
                return -np.dot(pesos, medio_ajustado) / np.sqrt(np.dot(pesos.T, np.dot(covarianza_rendimientos, pesos)))

            resultado = sco.minimize(objetivo_func, np.ones(len(media_rendimientos)) / len(media_rendimientos),
                                     method='SLSQP', bounds=[(0.035, 0.4) for _ in range(len(media_rendimientos))],
                                     constraints=restricciones)
            return resultado.x

        # Calcular pesos ajustados
        pesos_black_litterman = black_litterman_optimizar(media_rendimientos, covarianza_rendimientos, P, Q)

        # Visualización de expectativas
        st.subheader("Expectativas de Crecimiento para cada ETF")
        expectativas = {
            "LQD": "Esperamos buen crecimiento debido a su alta duración y exposición a sectores clave como banca y tecnología. Proyección: 8%.",
            "EMB": "Beneficio por bajas tasas de interés en mercados emergentes y exposición a bonos gubernamentales mexicanos. Proyección: 6.5%.",
            "SPY": "Exposición significativa a tecnología (31.66%) y políticas proteccionistas en EE.UU. Proyección: 12%.",
            "EMXC": "Diversificación en mercados emergentes excluyendo China, con alta exposición a tecnología. Proyección: 6%.",
            "IAU": "Commodities como cobertura inflacionaria; alta demanda de bancos centrales. Proyección: 5%."
        }

        for etf, expectativa in expectativas.items():
            st.write(f"**{etf}:** {expectativa}")

        # Mostrar restricciones del portafolio
        st.subheader("Restricciones del Portafolio")
        st.write("- Peso mínimo por activo: 3.5%")
        st.write("- Peso máximo por activo: 40%")

        # Gráfico de pastel
        fig = go.Figure(data=[
            go.Pie(labels=etfs, values=pesos_black_litterman, hoverinfo='label+percent')
        ])
        fig.update_layout(title="Distribución del Portafolio Ajustado - Black-Litterman")
        st.plotly_chart(fig)

        # Mostrar los pesos ajustados
        st.subheader("Pesos Ajustados del Portafolio")
        for etf, peso in zip(etfs, pesos_black_litterman):
            st.write(f"**{etf}:** {peso:.2%}")

    except Exception as e:
        st.error(f"Ocurrió un error al calcular el modelo Black-Litterman: {e}")

        st.write(data["estilo"])
        st.subheader("Costos")
        st.write(data["costos"])


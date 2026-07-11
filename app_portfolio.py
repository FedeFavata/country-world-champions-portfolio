# ============================================================
# APP STREAMLIT - OPTIMIZACION DE PORTAFOLIO
# ============================================================
# Para instalar:
# pip install streamlit yfinance pandas numpy scipy openpyxl matplotlib

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACION GENERAL
# ============================================================

st.set_page_config(
    page_title="Country World Champions ADR Portfolio",
    layout="wide"
)

# ============================================================
# PORTADA / PRESENTACION DEL PROYECTO
# ============================================================

st.markdown(
    """
    <div style="
        background-color:#0E1117;
        padding:35px;
        border-radius:18px;
        margin-bottom:25px;
        border:1px solid #30363D;
    ">
        <h1 style="color:white; margin-bottom:5px;">
            Country World Champions ADR Portfolio
        </h1>
        <h3 style="color:#AAB2BF; margin-top:0;">
            Construcción y optimización de una cartera internacional de líderes nacionales
        </h3>
        <hr style="border:0.5px solid #30363D;">
        <p style="color:#D0D7DE; font-size:18px;">
            <b>Grupo 2</b>
        </p>
        <p style="color:#D0D7DE; font-size:17px;">
            <b>Integrantes:</b> Antonucci, Favata, Manzini, Nestler y Sansone
        </p>
        <p style="color:#D0D7DE; font-size:16px; margin-top:18px;">
            Esta aplicación permite descargar precios históricos de activos, calcular retornos,
            esperanzas, volatilidades, correlaciones, covarianzas y optimizar una cartera
            maximizando el ratio de Sharpe.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "A continuación se ingresan los parámetros del análisis y se calculan los resultados del portfolio."
)

# ============================================================
# INPUTS DEL USUARIO
# ============================================================

st.sidebar.header("Parámetros del análisis")

fecha_inicio = st.sidebar.date_input(
    "Fecha inicio",
    value=pd.to_datetime("2015-01-01")
)

fecha_final = st.sidebar.date_input(
    "Fecha final",
    value=pd.to_datetime("2026-06-30")
)

tickers_input = st.sidebar.text_area(
    "Especies / tickers separados por coma",
    value="LVMUY, IDEXY, BUD, EQNR, HSBC, YPF, RHHBY, AFK"
)

rf_annual_pct = st.sidebar.number_input(
    "Tasa libre de riesgo anual (%)",
    value=4.50,
    step=0.10,
    format="%.2f"
)

rf_annual = rf_annual_pct / 100

max_weight = st.sidebar.number_input(
    "Peso máximo por activo",
    value=1.00,
    min_value=0.01,
    max_value=1.00,
    step=0.05,
    format="%.2f"
)

min_weight = st.sidebar.number_input(
    "Peso mínimo por activo",
    value=0.00,
    min_value=0.00,
    max_value=1.00,
    step=0.01,
    format="%.2f"
)

n_simulations = st.sidebar.number_input(
    "Cantidad de portfolios simulados",
    value=5000,
    min_value=500,
    max_value=50000,
    step=500
)

boton = st.sidebar.button("Calcular portfolio")

# ============================================================
# FUNCIONES
# ============================================================

def limpiar_tickers(tickers_input):
    tickers = [x.strip() for x in tickers_input.split(",")]
    tickers = [x for x in tickers if x != ""]
    return tickers


def descargar_precios(tickers, fecha_inicio, fecha_final):
    data = yf.download(
        tickers=tickers,
        start=fecha_inicio,
        end=fecha_final,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna(axis=1, how="all")
    prices.index = pd.to_datetime(prices.index)

    # Último precio disponible de cada mes.
    # Compatible con distintas versiones de pandas.
    try:
        prices_monthly = prices.resample("ME").last()
    except ValueError:
        prices_monthly = prices.resample("M").last()

    return prices_monthly

def calcular_metricas(returns_monthly):
    mu_monthly = returns_monthly.mean()
    mu_annual = mu_monthly * 12

    cov_monthly = returns_monthly.cov()
    cov_annual = cov_monthly * 12

    corr_matrix = returns_monthly.corr()

    vol_annual = returns_monthly.std() * np.sqrt(12)

    summary = pd.DataFrame({
        "Esperanza anual": mu_annual,
        "Volatilidad anual": vol_annual
    })

    return mu_annual, cov_annual, corr_matrix, summary


def portfolio_return(weights, mu):
    return np.dot(weights, mu)


def portfolio_volatility(weights, cov):
    return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))


def portfolio_sharpe(weights, mu, cov, rf_annual):
    ret = portfolio_return(weights, mu)
    vol = portfolio_volatility(weights, cov)

    if vol == 0:
        return np.nan

    return (ret - rf_annual) / vol


def negative_sharpe(weights, mu, cov, rf_annual):
    return -portfolio_sharpe(weights, mu, cov, rf_annual)


def optimizar_sharpe(mu, cov, rf_annual, min_weight, max_weight):
    n_assets = len(mu)

    constraints = ({
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    })

    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

    initial_weights = np.array([1 / n_assets] * n_assets)

    result = minimize(
        negative_sharpe,
        initial_weights,
        args=(mu, cov, rf_annual),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result


def limpiar_valores_chicos(x, limite=1e-6):
    if abs(x) < limite:
        return 0
    return x


# ============================================================
# CALCULO PRINCIPAL
# ============================================================

if boton:

    tickers = limpiar_tickers(tickers_input)

    if len(tickers) == 0:
        st.error("Tenés que ingresar al menos un ticker.")
        st.stop()

    if min_weight * len(tickers) > 1:
        st.error("El peso mínimo es demasiado alto para la cantidad de activos. Bajalo.")
        st.stop()

    if max_weight * len(tickers) < 1:
        st.error("El peso máximo es demasiado bajo para poder sumar 100%. Subilo.")
        st.stop()

    st.divider()

    st.subheader("Tickers seleccionados")
    st.write(tickers)

    # ========================================================
    # DESCARGA DE PRECIOS
    # ========================================================

    prices_monthly = descargar_precios(tickers, fecha_inicio, fecha_final)

    if prices_monthly.empty:
        st.error("No se pudieron descargar precios. Revisá los tickers.")
        st.stop()

    # Eliminar columnas sin datos suficientes
    min_obs = 12
    valid_columns = prices_monthly.columns[prices_monthly.notna().sum() >= min_obs]
    prices_monthly = prices_monthly[valid_columns]

    if prices_monthly.shape[1] == 0:
        st.error("Ningún ticker tiene suficientes datos.")
        st.stop()

    # ========================================================
    # RETORNOS
    # ========================================================

    returns_monthly = prices_monthly.pct_change().dropna(how="all")
    returns_monthly = returns_monthly.dropna(axis=1, how="all")

    # Para optimización uso filas completas
    returns_clean = returns_monthly.dropna()

    if returns_clean.empty:
        st.error("No hay suficientes retornos completos para optimizar.")
        st.stop()

    # ========================================================
    # METRICAS
    # ========================================================

    mu_annual, cov_annual, corr_matrix, summary = calcular_metricas(returns_clean)

    # ========================================================
    # OPTIMIZACION
    # ========================================================

    result = optimizar_sharpe(
        mu=mu_annual,
        cov=cov_annual,
        rf_annual=rf_annual,
        min_weight=min_weight,
        max_weight=max_weight
    )

    if not result.success:
        st.warning("La optimización no convergió perfectamente. Revisá restricciones o datos.")

    optimal_weights = result.x

    optimal_return = portfolio_return(optimal_weights, mu_annual)
    optimal_volatility = portfolio_volatility(optimal_weights, cov_annual)
    optimal_sharpe = portfolio_sharpe(optimal_weights, mu_annual, cov_annual, rf_annual)

    weights_df = pd.DataFrame({
        "Activo": returns_clean.columns,
        "Peso óptimo": optimal_weights,
        "Peso óptimo (%)": optimal_weights * 100
    })

    weights_df["Peso óptimo"] = weights_df["Peso óptimo"].apply(lambda x: limpiar_valores_chicos(x))
    weights_df["Peso óptimo (%)"] = weights_df["Peso óptimo (%)"].apply(lambda x: 0 if abs(x) < 1e-4 else x)

    weights_df = weights_df.sort_values("Peso óptimo", ascending=False)

    # ========================================================
    # OUTPUT PRINCIPAL
    # ========================================================

    st.header("Resultados del portfolio óptimo")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Retorno esperado anual", f"{optimal_return:.2%}")
    col2.metric("Volatilidad anual", f"{optimal_volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{optimal_sharpe:.3f}")
    col4.metric("Risk Free", f"{rf_annual:.2%}")

    st.subheader("Pesos óptimos")

    weights_show = weights_df[["Activo", "Peso óptimo (%)"]].copy()
    weights_show["Peso óptimo (%)"] = weights_show["Peso óptimo (%)"].map(lambda x: f"{x:.2f}%")

    st.dataframe(weights_show, use_container_width=True)

    # ========================================================
    # GRAFICO RIESGO - RETORNO
    # ========================================================

    st.subheader("Gráfico riesgo-retorno")

    results_sim = []

    for _ in range(int(n_simulations)):
        random_weights = np.random.random(len(mu_annual))
        random_weights = random_weights / np.sum(random_weights)

        ret_sim = portfolio_return(random_weights, mu_annual)
        vol_sim = portfolio_volatility(random_weights, cov_annual)
        sharpe_sim = (ret_sim - rf_annual) / vol_sim if vol_sim != 0 else np.nan

        results_sim.append([ret_sim, vol_sim, sharpe_sim])

    results_sim = pd.DataFrame(
        results_sim,
        columns=["Retorno", "Volatilidad", "Sharpe"]
    )

    asset_returns = mu_annual
    asset_vols = np.sqrt(np.diag(cov_annual))

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        results_sim["Volatilidad"],
        results_sim["Retorno"],
        c=results_sim["Sharpe"],
        cmap="viridis",
        alpha=0.35,
        s=12
    )

    ax.scatter(
        asset_vols,
        asset_returns,
        color="steelblue",
        s=80,
        label="Activos individuales"
    )

    for ticker, vol, ret in zip(mu_annual.index, asset_vols, asset_returns):
        ax.text(
            vol,
            ret,
            ticker,
            fontsize=9,
            ha="left",
            va="bottom"
        )

    ax.scatter(
        optimal_volatility,
        optimal_return,
        color="red",
        s=160,
        label="Portfolio óptimo"
    )

    ax.text(
        optimal_volatility,
        optimal_return,
        "  Óptimo",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="center"
    )

    ax.scatter(
        0,
        rf_annual,
        color="orange",
        s=100,
        label="Tasa libre de riesgo"
    )

    ax.text(
        0,
        rf_annual,
        "  Rf",
        fontsize=10,
        ha="left",
        va="center"
    )

    ax.plot(
        [0, optimal_volatility],
        [rf_annual, optimal_return],
        color="orange",
        linewidth=2,
        linestyle="--",
        label="Línea de asignación"
    )

    ax.set_xlabel("Desvío estándar anual")
    ax.set_ylabel("Retorno esperado anual")
    ax.set_title("Relación riesgo-retorno del portfolio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio")

    st.pyplot(fig)

    # ========================================================
    # TABLAS COMPLETAS
    # ========================================================

    st.subheader("Esperanzas y volatilidades anualizadas (%)")

    summary_show = summary * 100

    st.dataframe(
        summary_show.style.format("{:.2f}%"),
        use_container_width=True
    )

    st.subheader("Matriz de correlaciones")

    st.dataframe(
        corr_matrix.style.format("{:.4f}"),
        use_container_width=True
    )

    st.subheader("Matriz de covarianzas anuales (%)")

    cov_annual_show = cov_annual * 100

    st.dataframe(
        cov_annual_show.style.format("{:.4f}%"),
        use_container_width=True
    )

    st.subheader("Precios mensuales")

    st.dataframe(
        prices_monthly.round(2),
        use_container_width=True
    )

    st.subheader("Retornos mensuales (%)")

    returns_monthly_show = returns_monthly * 100

    st.dataframe(
        returns_monthly_show.style.format("{:.2f}%"),
        use_container_width=True
    )

    # ========================================================
    # DESCARGA EXCEL
    # ========================================================

    output_file = "resultado_portfolio.xlsx"

    portfolio_metrics = pd.DataFrame({
        "Metrica": [
            "Retorno esperado anual (%)",
            "Volatilidad anual (%)",
            "Sharpe Ratio",
            "Tasa libre de riesgo anual (%)"
        ],
        "Valor": [
            optimal_return * 100,
            optimal_volatility * 100,
            optimal_sharpe,
            rf_annual * 100
        ]
    })

    weights_excel = weights_df[["Activo", "Peso óptimo (%)"]].copy()

    summary_excel = summary * 100
    summary_excel.columns = ["Esperanza anual (%)", "Volatilidad anual (%)"]

    cov_annual_excel = cov_annual * 100
    returns_monthly_excel = returns_monthly * 100

    results_sim_excel = results_sim.copy()
    results_sim_excel["Retorno (%)"] = results_sim_excel["Retorno"] * 100
    results_sim_excel["Volatilidad (%)"] = results_sim_excel["Volatilidad"] * 100
    results_sim_excel = results_sim_excel[["Retorno (%)", "Volatilidad (%)", "Sharpe"]]

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        prices_monthly.to_excel(writer, sheet_name="Precios mensuales")
        returns_monthly_excel.to_excel(writer, sheet_name="Retornos mensuales %")
        summary_excel.to_excel(writer, sheet_name="Metricas activos %")
        corr_matrix.to_excel(writer, sheet_name="Correlaciones")
        cov_annual_excel.to_excel(writer, sheet_name="Covarianzas anuales %")
        weights_excel.to_excel(writer, sheet_name="Pesos optimos %", index=False)
        portfolio_metrics.to_excel(writer, sheet_name="Portfolio optimo", index=False)
        results_sim_excel.to_excel(writer, sheet_name="Portfolios simulados", index=False)

    with open(output_file, "rb") as file:
        st.download_button(
            label="Descargar resultados en Excel",
            data=file,
            file_name="resultado_portfolio.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    st.info(
        "Ingresá las fechas, tickers y parámetros en el panel izquierdo. "
        "Después tocá 'Calcular portfolio'."
    )

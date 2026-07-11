# ============================================================
# APP STREAMLIT - COUNTRY WORLD CHAMPIONS ADR PORTFOLIO
# ============================================================
# requirements.txt:
# streamlit
# yfinance
# pandas
# numpy
# scipy
# openpyxl
# matplotlib

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
    page_title="UTDT | Optimizador de Portafolio",
    layout="wide"
)

# ============================================================
# ESTILO UTDT
# ============================================================

UTDT_BLACK = "#0E0E0E"
UTDT_DARK = "#1A1A1A"
UTDT_YELLOW = "#F5A800"
UTDT_TURQUOISE = "#00C4B3"
UTDT_WHITE = "#FFFFFF"
UTDT_GRAY = "#D9D9D9"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #F7F7F7;
        }}

        section[data-testid="stSidebar"] {{
            background-color: {UTDT_BLACK};
        }}

        section[data-testid="stSidebar"] * {{
            color: white !important;
        }}

        div[data-testid="stMetric"] {{
            background-color: white;
            border: 1px solid #E5E5E5;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        }}

        h1, h2, h3 {{
            color: {UTDT_BLACK};
        }}

        .utdt-hero {{
            background: linear-gradient(135deg, {UTDT_BLACK} 0%, {UTDT_DARK} 70%);
            padding: 34px;
            border-radius: 20px;
            margin-bottom: 25px;
            border-left: 10px solid {UTDT_YELLOW};
            box-shadow: 0px 6px 18px rgba(0,0,0,0.18);
        }}

        .utdt-title {{
            color: white;
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 4px;
        }}

        .utdt-subtitle {{
            color: {UTDT_GRAY};
            font-size: 21px;
            margin-top: 0px;
            margin-bottom: 18px;
        }}

        .utdt-pill {{
            display: inline-block;
            background-color: {UTDT_YELLOW};
            color: {UTDT_BLACK};
            padding: 7px 14px;
            border-radius: 999px;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 10px;
        }}

        .utdt-pill-2 {{
            display: inline-block;
            background-color: {UTDT_TURQUOISE};
            color: {UTDT_BLACK};
            padding: 7px 14px;
            border-radius: 999px;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 10px;
        }}

        .utdt-description {{
            color: #EFEFEF;
            font-size: 16px;
            line-height: 1.55;
            max-width: 1050px;
        }}

        .utdt-section {{
            background-color: white;
            padding: 20px;
            border-radius: 14px;
            border: 1px solid #E6E6E6;
            margin-bottom: 18px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PORTADA
# ============================================================

col_logo, col_text = st.columns([1, 5])

with col_logo:
    try:
        st.image("utdt_logo.png", width=180)
    except Exception:
        st.markdown(
            f"""
            <div style="
                background-color:{UTDT_BLACK};
                padding:18px;
                border-radius:14px;
                border-left:8px solid {UTDT_YELLOW};
                color:white;
                text-align:center;
                font-weight:800;
                font-size:24px;
            ">
                UTDT
            </div>
            """,
            unsafe_allow_html=True
        )

with col_text:
    st.markdown(
        """
        <div class="utdt-hero">
            <div class="utdt-title">Optimizador de portafolios</div>
            <div class="utdt-subtitle">
                Construcción y optimización de una cartera de acciones
            </div>

            <span class="utdt-pill">Grupo 2</span>
            <span class="utdt-pill-2">Finanzas Personales</span>
            <span class="utdt-pill">Universidad Torcuato Di Tella</span>

            <p class="utdt-description">
                Integrantes: <b>Antonucci, Favata, Manzini, Nestler y Sansone</b>.
                La aplicación permite descargar precios históricos, calcular retornos,
                esperanzas, volatilidades, correlaciones, covarianzas, drawdowns y optimizar
                una cartera mediante máximo Sharpe o retorno objetivo.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="utdt-section">
        <b>Objetivo del análisis:</b> evaluar una estrategia global de inversión basada en empresas
        representativas de distintos países, usando activos comparables en USD y herramientas de
        optimización de portafolios.
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR - INPUTS
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

st.sidebar.divider()

st.sidebar.subheader("Restricciones de pesos")

allow_short = st.sidebar.checkbox(
    "Permitir short selling / pesos negativos",
    value=False
)

if allow_short:
    min_weight_pct = st.sidebar.number_input(
        "Peso mínimo por activo (%)",
        value=-30.00,
        min_value=-200.00,
        max_value=0.00,
        step=5.00,
        format="%.2f"
    )

    max_weight_pct = st.sidebar.number_input(
        "Peso máximo por activo (%)",
        value=100.00,
        min_value=1.00,
        max_value=300.00,
        step=5.00,
        format="%.2f"
    )
else:
    min_weight_pct = st.sidebar.number_input(
        "Peso mínimo por activo (%)",
        value=0.00,
        min_value=0.00,
        max_value=100.00,
        step=1.00,
        format="%.2f"
    )

    max_weight_pct = st.sidebar.number_input(
        "Peso máximo por activo (%)",
        value=100.00,
        min_value=1.00,
        max_value=100.00,
        step=5.00,
        format="%.2f"
    )

min_weight = min_weight_pct / 100
max_weight = max_weight_pct / 100

st.sidebar.divider()

st.sidebar.subheader("Método de optimización")

optimization_method = st.sidebar.radio(
    "Elegir método",
    [
        "Maximizar Sharpe",
        "Retorno objetivo"
    ]
)

target_return_pct = None
target_return = None

if optimization_method == "Retorno objetivo":
    target_return_pct = st.sidebar.number_input(
        "Retorno objetivo anual (%)",
        value=12.00,
        step=0.50,
        format="%.2f"
    )

    target_return = target_return_pct / 100

st.sidebar.divider()

st.sidebar.subheader("Benchmark")

use_benchmark = st.sidebar.checkbox(
    "Comparar contra benchmark",
    value=True
)

benchmark_ticker = st.sidebar.text_input(
    "Ticker benchmark",
    value="SPY"
)

st.sidebar.divider()

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
    try:
        data = yf.download(
            tickers=tickers,
            start=fecha_inicio,
            end=fecha_final,
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                prices = data["Close"]
            else:
                return pd.DataFrame()
        else:
            if "Close" in data.columns:
                prices = data[["Close"]]
                prices.columns = tickers
            else:
                return pd.DataFrame()

        prices = prices.dropna(axis=1, how="all")

        if prices.empty:
            return pd.DataFrame()

        prices.index = pd.to_datetime(prices.index)

        try:
            prices_monthly = prices.resample("ME").last()
        except ValueError:
            prices_monthly = prices.resample("M").last()

        return prices_monthly

    except Exception as e:
        st.error("Ocurrió un error al descargar precios.")
        st.code(str(e))
        return pd.DataFrame()


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


def portfolio_variance(weights, cov):
    return np.dot(weights.T, np.dot(cov, weights))


def optimizar_portfolio(
    mu,
    cov,
    rf_annual,
    min_weight,
    max_weight,
    optimization_method,
    target_return=None
):
    n_assets = len(mu)

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        }
    ]

    if optimization_method == "Retorno objetivo":
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: portfolio_return(weights, mu) - target_return
            }
        )

    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    initial_weights = np.array([1 / n_assets] * n_assets)

    if optimization_method == "Maximizar Sharpe":
        objective = negative_sharpe
        args = (mu, cov, rf_annual)
    else:
        objective = portfolio_volatility
        args = (cov,)

    result = minimize(
        objective,
        initial_weights,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result


def calcular_drawdown(returns):
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown


def generar_portfolios_simulados(mu, cov, rf_annual, min_weight, max_weight, n_simulations):
    n_assets = len(mu)
    results = []
    weights_list = []

    max_attempts = int(n_simulations) * 40
    attempts = 0

    while len(results) < int(n_simulations) and attempts < max_attempts:
        attempts += 1

        if min_weight < 0:
            w = np.random.uniform(min_weight, max_weight, n_assets)

            if np.sum(w) == 0:
                continue

            w = w / np.sum(w)

            if np.any(w < min_weight - 1e-8) or np.any(w > max_weight + 1e-8):
                continue
        else:
            w = np.random.random(n_assets)
            w = w / np.sum(w)

            if np.any(w < min_weight - 1e-8) or np.any(w > max_weight + 1e-8):
                continue

        ret = portfolio_return(w, mu)
        vol = portfolio_volatility(w, cov)
        sharpe = (ret - rf_annual) / vol if vol != 0 else np.nan

        results.append([ret, vol, sharpe])
        weights_list.append(w)

    results_df = pd.DataFrame(
        results,
        columns=["Retorno", "Volatilidad", "Sharpe"]
    )

    weights_sim_df = pd.DataFrame(
        weights_list,
        columns=mu.index
    )

    return results_df, weights_sim_df


def limpiar_valores_chicos(x, limite=1e-8):
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

    result = optimizar_portfolio(
        mu=mu_annual,
        cov=cov_annual,
        rf_annual=rf_annual,
        min_weight=min_weight,
        max_weight=max_weight,
        optimization_method=optimization_method,
        target_return=target_return
    )

    if not result.success:
        st.error("La optimización no encontró una solución factible. Probá relajar restricciones o bajar el retorno objetivo.")
        st.code(result.message)
        st.stop()

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
    weights_df["Peso óptimo (%)"] = weights_df["Peso óptimo (%)"].apply(lambda x: 0 if abs(x) < 1e-6 else x)

    weights_df = weights_df.sort_values("Peso óptimo", ascending=False)

    long_exposure = weights_df.loc[weights_df["Peso óptimo"] > 0, "Peso óptimo"].sum()
    short_exposure = weights_df.loc[weights_df["Peso óptimo"] < 0, "Peso óptimo"].sum()
    net_exposure = weights_df["Peso óptimo"].sum()

    # Portfolio histórico
    portfolio_returns = returns_clean.dot(optimal_weights)
    portfolio_wealth = (1 + portfolio_returns).cumprod() * 100
    portfolio_drawdown, portfolio_max_drawdown = calcular_drawdown(portfolio_returns)

    equal_weights = np.array([1 / len(returns_clean.columns)] * len(returns_clean.columns))
    equal_returns = returns_clean.dot(equal_weights)
    equal_wealth = (1 + equal_returns).cumprod() * 100
    equal_drawdown, equal_max_drawdown = calcular_drawdown(equal_returns)

    # Benchmark
    benchmark_returns = None
    benchmark_wealth = None
    benchmark_max_drawdown = None

    if use_benchmark and benchmark_ticker.strip() != "":
        benchmark_prices = descargar_precios([benchmark_ticker.strip()], fecha_inicio, fecha_final)

        if not benchmark_prices.empty:
            benchmark_returns_raw = benchmark_prices.pct_change().dropna()

            if benchmark_returns_raw.shape[1] > 0:
                benchmark_returns = benchmark_returns_raw.iloc[:, 0]
                benchmark_returns = benchmark_returns.loc[benchmark_returns.index.intersection(portfolio_returns.index)]
                benchmark_wealth = (1 + benchmark_returns).cumprod() * 100
                _, benchmark_max_drawdown = calcular_drawdown(benchmark_returns)

    # ========================================================
    # OUTPUT PRINCIPAL
    # ========================================================

    st.header("Resultados del portfolio óptimo")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Retorno esperado anual", f"{optimal_return:.2%}")
    col2.metric("Volatilidad anual", f"{optimal_volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{optimal_sharpe:.3f}")
    col4.metric("Max Drawdown", f"{portfolio_max_drawdown:.2%}")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Risk Free", f"{rf_annual:.2%}")
    col6.metric("Exposición Long", f"{long_exposure:.2%}")
    col7.metric("Exposición Short", f"{short_exposure:.2%}")
    col8.metric("Exposición Neta", f"{net_exposure:.2%}")

    st.caption(f"Método utilizado: **{optimization_method}**")

    if optimization_method == "Retorno objetivo":
        st.caption(f"Retorno objetivo anual cargado: **{target_return:.2%}**")

    # ========================================================
    # PESOS OPTIMOS
    # ========================================================

    st.subheader("Pesos óptimos")

    weights_show = weights_df[["Activo", "Peso óptimo (%)"]].copy()
    weights_show["Peso óptimo (%)"] = weights_show["Peso óptimo (%)"].map(lambda x: f"{x:.2f}%")

    st.dataframe(weights_show, use_container_width=True)

    # Pie chart
    st.subheader("Distribución de pesos óptimos")

    pie_df = weights_df.copy()
    pie_df["Peso abs"] = pie_df["Peso óptimo"].abs()
    pie_df = pie_df[pie_df["Peso abs"] > 0.0001]

    fig_pie, ax_pie = plt.subplots(figsize=(8, 6))

    ax_pie.pie(
        pie_df["Peso abs"],
        labels=pie_df["Activo"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax_pie.set_title("Participación por peso absoluto")
    st.pyplot(fig_pie)

    if allow_short:
        st.caption("Nota: como hay short selling, el gráfico muestra pesos absolutos para visualizar la magnitud de cada posición.")

    # ========================================================
    # GRAFICO PRECIOS HISTORICOS BASE 100
    # ========================================================

    st.subheader("Evolución histórica de precios - Base 100")

    prices_base100 = prices_monthly.copy()
    prices_base100 = prices_base100 / prices_base100.dropna().iloc[0] * 100

    st.line_chart(prices_base100)

    # ========================================================
    # WEALTH INDEX PORTFOLIO VS EQUAL WEIGHT VS BENCHMARK
    # ========================================================

    st.subheader("Evolución de USD 100 invertidos")

    wealth_df = pd.DataFrame({
        "Portfolio óptimo": portfolio_wealth,
        "Portfolio igual ponderado": equal_wealth
    })

    if benchmark_wealth is not None:
        wealth_df[f"Benchmark {benchmark_ticker.strip()}"] = benchmark_wealth

    st.line_chart(wealth_df)

    # ========================================================
    # DRAWDOWN
    # ========================================================

    st.subheader("Drawdown histórico")

    drawdown_df = pd.DataFrame({
        "Portfolio óptimo": portfolio_drawdown * 100,
        "Portfolio igual ponderado": equal_drawdown * 100
    })

    if benchmark_returns is not None:
        benchmark_drawdown, _ = calcular_drawdown(benchmark_returns)
        drawdown_df[f"Benchmark {benchmark_ticker.strip()}"] = benchmark_drawdown * 100

    st.line_chart(drawdown_df)

    if benchmark_max_drawdown is not None:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Max DD Portfolio óptimo", f"{portfolio_max_drawdown:.2%}")
        col_b.metric("Max DD Igual ponderado", f"{equal_max_drawdown:.2%}")
        col_c.metric(f"Max DD {benchmark_ticker.strip()}", f"{benchmark_max_drawdown:.2%}")
    else:
        col_a, col_b = st.columns(2)
        col_a.metric("Max DD Portfolio óptimo", f"{portfolio_max_drawdown:.2%}")
        col_b.metric("Max DD Igual ponderado", f"{equal_max_drawdown:.2%}")

    # ========================================================
    # GRAFICO RIESGO - RETORNO
    # ========================================================

    st.subheader("Gráfico riesgo-retorno")

    results_sim, weights_sim = generar_portfolios_simulados(
        mu=mu_annual,
        cov=cov_annual,
        rf_annual=rf_annual,
        min_weight=min_weight,
        max_weight=max_weight,
        n_simulations=n_simulations
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
    # TABLAS
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

    # Heatmap correlaciones
    st.subheader("Heatmap de correlaciones")

    fig_heat, ax_heat = plt.subplots(figsize=(9, 7))

    im = ax_heat.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    ax_heat.set_xticks(np.arange(len(corr_matrix.columns)))
    ax_heat.set_yticks(np.arange(len(corr_matrix.index)))

    ax_heat.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax_heat.set_yticklabels(corr_matrix.index)

    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax_heat.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    ax_heat.set_title("Matriz de correlaciones")
    fig_heat.colorbar(im, ax=ax_heat)
    st.pyplot(fig_heat)

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
            "Metodo de optimizacion",
            "Retorno objetivo anual (%)",
            "Retorno esperado anual (%)",
            "Volatilidad anual (%)",
            "Sharpe Ratio",
            "Tasa libre de riesgo anual (%)",
            "Max Drawdown (%)",
            "Exposicion Long (%)",
            "Exposicion Short (%)",
            "Exposicion Neta (%)"
        ],
        "Valor": [
            optimization_method,
            target_return * 100 if target_return is not None else "",
            optimal_return * 100,
            optimal_volatility * 100,
            optimal_sharpe,
            rf_annual * 100,
            portfolio_max_drawdown * 100,
            long_exposure * 100,
            short_exposure * 100,
            net_exposure * 100
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

    wealth_excel = wealth_df.copy()
    drawdown_excel = drawdown_df.copy()

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        prices_monthly.to_excel(writer, sheet_name="Precios mensuales")
        prices_base100.to_excel(writer, sheet_name="Precios Base 100")
        returns_monthly_excel.to_excel(writer, sheet_name="Retornos mensuales %")
        summary_excel.to_excel(writer, sheet_name="Metricas activos %")
        corr_matrix.to_excel(writer, sheet_name="Correlaciones")
        cov_annual_excel.to_excel(writer, sheet_name="Covarianzas anuales %")
        weights_excel.to_excel(writer, sheet_name="Pesos optimos %", index=False)
        portfolio_metrics.to_excel(writer, sheet_name="Portfolio optimo", index=False)
        results_sim_excel.to_excel(writer, sheet_name="Portfolios simulados", index=False)
        wealth_excel.to_excel(writer, sheet_name="Wealth Index")
        drawdown_excel.to_excel(writer, sheet_name="Drawdown %")

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

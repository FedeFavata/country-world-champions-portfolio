# ============================================================
# APP STREAMLIT - OPTIMIZADOR DE PORTAFOLIOS
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
    page_title="UTDT | Optimizador de Portafolios",
    layout="wide"
)

# ============================================================
# PORTADA MINIMALISTA
# ============================================================

st.caption("UNIVERSIDAD TORCUATO DI TELLA")
st.title("Optimizador de Portafolios")
st.subheader("Construcción y optimización de una cartera de acciones")

st.markdown(
    """
**Grupo 2**  
**Integrantes:** Antonucci · Favata · Manzini · Nestler · Sansone  

Esta aplicación permite descargar precios históricos, calcular retornos, volatilidades,
correlaciones, covarianzas, drawdowns y optimizar una cartera utilizando máximo Sharpe
o retorno objetivo.
"""
)

st.divider()

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
    value="LVMUY, IDEXY, BUD, EQNR, HSBC, YPF, RHHBY"
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
    ["Maximizar Sharpe", "Retorno objetivo"]
)

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

n_simulations = st.sidebar.number_input(
    "Cantidad de portfolios simulados",
    value=2000,
    min_value=500,
    max_value=20000,
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


@st.cache_data(ttl=3600)
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

        if data is None:
            return pd.DataFrame()

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" not in data.columns.get_level_values(0):
                return pd.DataFrame()

            prices = data["Close"]

        else:
            if "Close" not in data.columns:
                return pd.DataFrame()

            prices = data[["Close"]]
            prices.columns = tickers

        prices = prices.dropna(axis=1, how="all")

        if prices.empty:
            return pd.DataFrame()

        prices.index = pd.to_datetime(prices.index)

        try:
            prices_monthly = prices.resample("ME").last()
        except Exception:
            prices_monthly = prices.resample("M").last()

        return prices_monthly

    except Exception as e:
        st.error("Ocurrió un error al descargar precios.")
        st.exception(e)
        return pd.DataFrame()


def calcular_metricas(returns_monthly):
    mu_annual = returns_monthly.mean() * 12
    cov_annual = returns_monthly.cov() * 12
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


def limpiar_valores_chicos(x, limite=1e-8):
    if abs(x) < limite:
        return 0
    return x


def calcular_drawdown(returns):
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown


def generar_portfolios_simulados(mu, cov, rf_annual, min_weight, max_weight, n_simulations):
    results = []
    attempts = 0
    max_attempts = int(n_simulations) * 40
    n_assets = len(mu)

    while len(results) < int(n_simulations) and attempts < max_attempts:
        attempts += 1

        if min_weight < 0:
            weights = np.random.uniform(min_weight, max_weight, n_assets)

            if np.sum(weights) == 0:
                continue

            weights = weights / np.sum(weights)

            if np.any(weights < min_weight) or np.any(weights > max_weight):
                continue

        else:
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)

            if np.any(weights < min_weight) or np.any(weights > max_weight):
                continue

        ret = portfolio_return(weights, mu)
        vol = portfolio_volatility(weights, cov)
        sharpe = (ret - rf_annual) / vol if vol != 0 else np.nan

        results.append([ret, vol, sharpe])

    return pd.DataFrame(results, columns=["Retorno", "Volatilidad", "Sharpe"])


# ============================================================
# CALCULO PRINCIPAL
# ============================================================

if boton:

    try:

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

        st.subheader("Tickers seleccionados")
        st.write(tickers)

        # ========================================================
        # DESCARGA DE PRECIOS
        # ========================================================

        with st.spinner("Descargando precios y calculando retornos..."):
            prices_monthly = descargar_precios(tickers, fecha_inicio, fecha_final)

        if prices_monthly is None or prices_monthly.empty:
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
        returns_clean = returns_monthly.dropna()

        if returns_clean.empty:
            st.error("No hay suficientes retornos completos para optimizar.")
            st.stop()

        if len(returns_clean) < 12:
            st.error("No hay suficientes observaciones para optimizar. Probá ampliar el período o cambiar los activos.")
            st.stop()

        # ========================================================
        # METRICAS
        # ========================================================

        mu_annual, cov_annual, corr_matrix, summary = calcular_metricas(returns_clean)

        # ========================================================
        # OPTIMIZACION
        # ========================================================

        with st.spinner("Optimizando portfolio..."):
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
            st.code(str(result.message))
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

        portfolio_returns = returns_clean.dot(optimal_weights)
        portfolio_wealth = (1 + portfolio_returns).cumprod() * 100
        portfolio_drawdown, portfolio_max_drawdown = calcular_drawdown(portfolio_returns)

        # ========================================================
        # RESULTADOS PRINCIPALES
        # ========================================================

        st.header("Resultados")

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

        st.divider()

        # ========================================================
        # PESOS
        # ========================================================

        st.subheader("Composición de la cartera")

        weights_show = weights_df[["Activo", "Peso óptimo (%)"]].copy()
        weights_show["Peso óptimo (%)"] = weights_show["Peso óptimo (%)"].map(lambda x: f"{x:.2f}%")

        st.dataframe(weights_show, use_container_width=True)

        fig_pie, ax_pie = plt.subplots(figsize=(7, 5))

        pie_df = weights_df.copy()
        pie_df["Peso absoluto"] = pie_df["Peso óptimo"].abs()
        pie_df = pie_df[pie_df["Peso absoluto"] > 0.0001]

        if not pie_df.empty:
            ax_pie.pie(
                pie_df["Peso absoluto"],
                labels=pie_df["Activo"],
                autopct="%1.1f%%",
                startangle=90
            )
            ax_pie.set_title("Pesos óptimos por activo")
            st.pyplot(fig_pie)

        if allow_short:
            st.caption("Nota: como se permite short selling, el gráfico muestra pesos absolutos.")

        # ========================================================
        # PRECIOS BASE 100
        # ========================================================

        st.subheader("Evolución histórica de precios - Base 100")

        prices_base100 = prices_monthly.copy()

        for col in prices_base100.columns:

            serie = prices_base100[col].dropna()

            if len(serie) == 0:
                continue

            first_valid = serie.iloc[0]

            if pd.isna(first_valid):
                continue

            if first_valid == 0:
                continue

            prices_base100[col] = (
                prices_base100[col] /
                first_valid
            ) * 100

        st.line_chart(prices_base100)

        # ========================================================
        # WEALTH INDEX
        # ========================================================

        st.subheader("Evolución de USD 100 invertidos")

        wealth_df = pd.DataFrame({
            "Portfolio óptimo": portfolio_wealth
        })

        st.line_chart(wealth_df)

        # ========================================================
        # DRAWDOWN
        # ========================================================

        st.subheader("Drawdown histórico del portfolio")

        drawdown_df = pd.DataFrame({
            "Portfolio óptimo": portfolio_drawdown * 100
        })

        st.line_chart(drawdown_df)

        # ========================================================
        # FRONTERA
        # ========================================================

        st.subheader("Frontera eficiente simulada")

        results_sim = generar_portfolios_simulados(
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

        if not results_sim.empty:
            scatter = ax.scatter(
                results_sim["Volatilidad"],
                results_sim["Retorno"],
                c=results_sim["Sharpe"],
                cmap="viridis",
                alpha=0.35,
                s=12
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Sharpe Ratio")

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
        ax.set_title("Relación riesgo-retorno")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
        ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")

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

        if not results_sim_excel.empty:
            results_sim_excel["Retorno (%)"] = results_sim_excel["Retorno"] * 100
            results_sim_excel["Volatilidad (%)"] = results_sim_excel["Volatilidad"] * 100
            results_sim_excel = results_sim_excel[["Retorno (%)", "Volatilidad (%)", "Sharpe"]]

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
            wealth_df.to_excel(writer, sheet_name="Wealth Index")
            drawdown_df.to_excel(writer, sheet_name="Drawdown %")

        with open(output_file, "rb") as file:
            st.download_button(
                label="Descargar resultados en Excel",
                data=file,
                file_name="resultado_portfolio.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error("Error durante la ejecución")
        st.exception(e)

else:
    st.info(
        "Ingresá los parámetros en el panel izquierdo y luego tocá **Calcular portfolio**."
    )

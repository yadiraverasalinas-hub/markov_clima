import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


ESTADOS = ["N", "PN", "PS", "S"]

NOMBRES_ESTADOS = {
    "N": "Nublado",
    "PN": "Parcialmente Nublado",
    "PS": "Parcialmente Soleado",
    "S": "Soleado",
}

COLORES_ESTADOS = {
    "N": "#607D8B",
    "PN": "#90A4AE",
    "PS": "#FFD54F",
    "S": "#FF8F00",
}


def _generate_random_sequence(n: int, rng: np.random.Generator) -> list[str]:
    pesos = np.array([0.35, 0.30, 0.20, 0.15], dtype=float)
    pesos = pesos / pesos.sum()
    seq = rng.choice(ESTADOS, size=int(n), p=pesos)
    return [str(s) for s in seq]


def _estimate_transition_from_sequence(secuencia: list[str], alpha: float = 1.0) -> pd.DataFrame:
    # Convención usada en la app:
    #   columnas = estado actual
    #   filas    = siguiente estado
    # por lo tanto cada columna debe sumar 1.
    idx = {s: i for i, s in enumerate(ESTADOS)}
    counts = np.zeros((len(ESTADOS), len(ESTADOS)), dtype=float)

    for a, b in zip(secuencia[:-1], secuencia[1:]):
        j = idx[a]  # estado actual (columna)
        i = idx[b]  # siguiente estado (fila)
        counts[i, j] += 1.0

    counts = counts + float(alpha)
    df = pd.DataFrame(counts, index=ESTADOS, columns=ESTADOS)
    return _normalize_columns(df)


def _random_transition_df(n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    base_seq = _generate_random_sequence(n=int(n_samples), rng=rng)
    return _estimate_transition_from_sequence(base_seq)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza cada columna para que sume 1.0 (si una columna suma 0, se deja igual)
    arr = df.to_numpy(dtype=float)
    col_sums = arr.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(col_sums != 0, arr / col_sums, arr)
    out = pd.DataFrame(norm, index=df.index, columns=df.columns)
    for col in out.columns:
        if out[col].sum() != 0:
            out.loc[out.index[-1], col] = 1.0 - float(out.loc[out.index[:-1], col].sum())
    return out


def _validate_transition_matrix(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    if list(df.index) != ESTADOS or list(df.columns) != ESTADOS:
        errors.append("La matriz debe tener filas y columnas en el orden: N, PN, PS, S.")
        return errors

    arr = df.to_numpy(dtype=float)

    if np.any(np.isnan(arr)):
        errors.append("La matriz contiene valores vacíos (NaN).")

    if np.any(arr < 0):
        errors.append("La matriz contiene probabilidades negativas.")

    col_sums = arr.sum(axis=0)
    if not np.allclose(col_sums, 1.0, atol=1e-6):
        errors.append(
            "Cada columna de la matriz debe sumar 1.0 (o usa la opción de normalizar automáticamente)."
        )

    return errors


def simulate_markov_chain(
    n_dias: int,
    estado_inicial: str,
    transition_df: pd.DataFrame,
    rng: np.random.Generator,
) -> list[str]:
    estados = ESTADOS
    state_to_idx = {s: i for i, s in enumerate(estados)}

    P = transition_df.to_numpy(dtype=float)

    current = estado_inicial
    seq = [current]

    for _ in range(n_dias - 1):
        i = state_to_idx[current]
        next_state = rng.choice(estados, p=P[:, i])
        seq.append(str(next_state))
        current = str(next_state)

    return seq


def build_results_df(secuencia: list[str]) -> pd.DataFrame:
    n = len(secuencia)
    return pd.DataFrame(
        {
            "Día": range(1, n + 1),
            "Código": secuencia,
            "Estado Climático": [NOMBRES_ESTADOS[s] for s in secuencia],
        }
    ).set_index("Día")


def build_frequency_df(secuencia: list[str]) -> pd.DataFrame:
    s = pd.Series(secuencia, name="Código")
    counts = s.value_counts().reindex(ESTADOS, fill_value=0)
    df = counts.rename("Días").to_frame()
    df["Porcentaje"] = (df["Días"] / df["Días"].sum()) * 100
    df["Estado"] = [NOMBRES_ESTADOS[c] for c in df.index]
    df["Color"] = [COLORES_ESTADOS[c] for c in df.index]
    return df.reset_index(drop=False).rename(columns={"index": "Código"})


st.set_page_config(
    page_title="Simulador Climático — Cadena de Markov",
    page_icon="☁️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2.5rem; }
      .stMetric { background: #0b1220; border: 1px solid rgba(255,255,255,.08); padding: 0.75rem; border-radius: 0.75rem; }
      div[data-testid='stDataFrame'] { border-radius: 0.75rem; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Simulador Climático con Cadenas de Markov")
st.caption("Genera una secuencia de estados: N, PN, PS, S — con matriz de transición configurable.")

with st.sidebar:
    st.header("Parámetros")

    n_dias = st.slider("Número de días", min_value=60, max_value=90, value=75, step=1)

    estado_inicial = st.selectbox(
        "Estado inicial",
        options=ESTADOS,
        format_func=lambda x: f"{x} — {NOMBRES_ESTADOS[x]}",
        index=0,
    )

    st.subheader("Semilla")
    usar_semilla = st.checkbox("Usar semilla fija (reproducible)", value=True)


seed_mode = "fixed" if usar_semilla else "random"
last_seed_mode = st.session_state.get("transition_seed_mode")

if ("transition_df" not in st.session_state) or (last_seed_mode != seed_mode):
    seed_value_init = 2026 if usar_semilla else None
    rng_init = np.random.default_rng(seed_value_init)
    st.session_state.transition_df = _random_transition_df(n_samples=max(200, n_dias), rng=rng_init)
    st.session_state.transition_seed_mode = seed_mode


transition_df = st.session_state.transition_df.copy()
transition_df = _normalize_columns(transition_df)

errors = _validate_transition_matrix(transition_df)
if errors:
    for e in errors:
        st.error(e)
    st.stop()

seed_value = 2026 if usar_semilla else None
rng = np.random.default_rng(seed_value)

colA, colB, colC, colD = st.columns(4)
colA.metric("Días", n_dias)
colB.metric("Estado inicial", f"{estado_inicial} ({NOMBRES_ESTADOS[estado_inicial]})")
colC.metric("Semilla", "Fija" if usar_semilla else "Aleatoria")
colD.metric("Estados", ", ".join(ESTADOS))

st.divider()

if st.button("Generar secuencia", type="primary"):
    if not usar_semilla:
        st.session_state.transition_df = _random_transition_df(
            n_samples=max(200, n_dias),
            rng=np.random.default_rng(),
        )
        transition_df = _normalize_columns(st.session_state.transition_df.copy())

    secuencia = simulate_markov_chain(
        n_dias=n_dias,
        estado_inicial=estado_inicial,
        transition_df=transition_df,
        rng=rng,
    )
    st.session_state["last_sequence"] = secuencia

secuencia = st.session_state.get("last_sequence")

if secuencia is None:
    st.info("Configura los parámetros y haz clic en **Generar secuencia**.")
    st.stop()

results_df = build_results_df(secuencia)
freq_df = build_frequency_df(secuencia)

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Tabla de resultados")
    st.dataframe(results_df, use_container_width=True, height=520)

with right:
    st.subheader("Distribución de frecuencias")

    fig = px.bar(
        freq_df,
        x="Estado",
        y="Días",
        color="Código",
        color_discrete_map=COLORES_ESTADOS,
        text=freq_df["Porcentaje"].map(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Número de días",
        xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        freq_df[["Código", "Estado", "Días", "Porcentaje"]].assign(
            Porcentaje=lambda d: d["Porcentaje"].map(lambda v: f"{v:.2f}%")
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

with st.expander("Ver matriz de transición estimada (usada en la simulación)", expanded=False):
    st.caption("Columnas = estado actual, filas = siguiente estado. Cada columna suma 1.")
    st.dataframe(transition_df.style.format("{:.3f}"), use_container_width=True)
    col_sums = transition_df.sum(axis=0)
    st.dataframe(
        pd.DataFrame({"Suma de columna": col_sums}).T.style.format("{:.3f}"),
        use_container_width=True,
    )

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


def _default_transition_df() -> pd.DataFrame:
    # Matriz por defecto (filas = estado actual, columnas = siguiente estado)
    # Diseñada para reflejar persistencia climática moderada.
    data = np.array(
        [
            [0.55, 0.25, 0.15, 0.05],  # N
            [0.25, 0.40, 0.25, 0.10],  # PN
            [0.10, 0.25, 0.40, 0.25],  # PS
            [0.05, 0.10, 0.25, 0.60],  # S
        ],
        dtype=float,
    )
    return pd.DataFrame(data, index=ESTADOS, columns=ESTADOS)


def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliza cada fila para que sume 1.0 (si una fila suma 0, se deja igual)
    arr = df.to_numpy(dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(row_sums != 0, arr / row_sums, arr)
    return pd.DataFrame(norm, index=df.index, columns=df.columns)


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

    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        errors.append(
            "Cada fila de la matriz debe sumar 1.0 (o usa la opción de normalizar automáticamente)."
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
        next_state = rng.choice(estados, p=P[i])
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

    st.subheader("Matriz de transición")
    st.write("Filas = estado actual, columnas = siguiente estado")

    normalize = st.checkbox("Normalizar filas automáticamente", value=True)

    if "transition_df" not in st.session_state:
        st.session_state.transition_df = _default_transition_df()

    edited_df = st.data_editor(
        st.session_state.transition_df,
        use_container_width=True,
        num_rows="fixed",
        key="transition_editor",
    )

    st.session_state.transition_df = edited_df

    if st.button("Restaurar matriz por defecto", use_container_width=True):
        st.session_state.transition_df = _default_transition_df()
        st.rerun()


transition_df = st.session_state.transition_df.copy()
if normalize:
    transition_df = _normalize_rows(transition_df)

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

with st.expander("Ver matriz de transición usada (tras normalización)", expanded=False):
    st.dataframe(transition_df.style.format("{:.4f}"), use_container_width=True)

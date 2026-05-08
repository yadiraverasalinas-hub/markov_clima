import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# MÓDULO 1: DEFINICIÓN DE ESTADOS DEL SISTEMA
# ============================================================

ESTADOS = ['N', 'PN', 'PS', 'S']

NOMBRES_ESTADOS = {
    'N' : 'Nublado',
    'PN': 'Parcialmente Nublado',
    'PS': 'Parcialmente Soleado',
    'S' : 'Soleado'
}

COLORES_ESTADOS = {
    'N' : '#607D8B',
    'PN': '#90A4AE',
    'PS': '#FFD54F',
    'S' : '#FF8F00'
}

INDICE_ESTADO = {}
for i, estado in enumerate(ESTADOS):
    INDICE_ESTADO[estado] = i

# ============================================================
# MÓDULO 2: GENERACIÓN DE DATOS CLIMÁTICOS SIMULADOS
# ============================================================

def generar_secuencia_climatica(n_dias: int, usar_semilla: bool) -> list[str]:
    """Genera una secuencia aleatoria de estados climáticos.

    - Usa los mismos pesos de tu código original.
    - Si `usar_semilla` es True, fija la semilla para reproducibilidad.
    """
    if usar_semilla:
        seed = 2026
    else:
        seed = None

    rng = np.random.default_rng(seed)

    # Pesos: N, PN, PS, S
    pesos = np.array([0.35, 0.30, 0.20, 0.15], dtype=float)
    pesos = pesos / pesos.sum()

    seq = rng.choice(ESTADOS, size=int(n_dias), p=pesos)

    # Convertir a lista de strings (por si numpy devuelve tipos distintos)
    secuencia = []
    for s in seq:
        secuencia.append(str(s))

    return secuencia


def construir_matriz_transicion_df(secuencia: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve la matriz de transición A y la matriz de conteos como DataFrames."""
    A, conteo = construir_matriz_transicion(secuencia)

    A_df = pd.DataFrame(A, index=ESTADOS, columns=ESTADOS)
    conteo_df = pd.DataFrame(conteo, index=ESTADOS, columns=ESTADOS)

    return A_df, conteo_df


def build_results_df(secuencia: list[str]) -> pd.DataFrame:
    """Tabla día a día: código del estado y nombre del estado."""
    dias = []
    codigos = []
    nombres = []

    for i, codigo in enumerate(secuencia):
        dias.append(i + 1)
        codigos.append(codigo)
        nombres.append(NOMBRES_ESTADOS[codigo])

    df = pd.DataFrame({"Día": dias, "Código": codigos, "Estado Climático": nombres})
    df = df.set_index("Día")
    return df


def build_frequency_df(secuencia: list[str]) -> pd.DataFrame:
    """Devuelve frecuencia por estado (días y porcentaje)."""
    conteo = Counter(secuencia)
    total = max(1, len(secuencia))

    codigos = []
    estados = []
    dias = []
    porcentajes = []

    for codigo in ESTADOS:
        d = int(conteo.get(codigo, 0))
        p = (d / total) * 100

        codigos.append(codigo)
        estados.append(NOMBRES_ESTADOS[codigo])
        dias.append(d)
        porcentajes.append(p)

    df = pd.DataFrame(
        {
            "Código": codigos,
            "Estado": estados,
            "Días": dias,
            "Porcentaje": porcentajes,
        }
    )
    return df


def crear_vector_estado(estado_actual: str) -> np.ndarray:
    """Crea el vector u0 con 1 en el estado actual y 0 en los demás."""
    u0 = np.zeros(len(ESTADOS))
    u0[INDICE_ESTADO[estado_actual]] = 1.0
    return u0


def predecir_estados_futuros(matriz: np.ndarray, estado_actual: str, pasos: int = 4) -> dict:
    """Aplica u_{k+1} = A @ u_k y guarda cada vector n+1, n+2, ..., n+pasos."""
    u = crear_vector_estado(estado_actual)
    predicciones = {}
    for paso in range(1, pasos + 1):
        u = matriz @ u
        predicciones[f"n+{paso}"] = u.copy()
    return predicciones


# ============================================================
# MÓDULO 3: CONSTRUCCIÓN DE LA MATRIZ DE TRANSICIÓN
# ============================================================

def construir_matriz_transicion(secuencia: list) -> tuple:
    """
    Construye la matriz de transición de la cadena de Márkov
    a partir de una secuencia de estados climáticos observados.

    Parámetros:
    -----------
    secuencia : list
        Lista de estados climáticos observados.

    Retorna:
    --------
    np.ndarray : Matriz de transición de tamaño 4x4.
    """
    n      = len(ESTADOS)
    conteo = np.zeros((n, n))

    for dia in range(len(secuencia) - 1):
        j = INDICE_ESTADO[secuencia[dia]]       # estado actual  → columna
        i = INDICE_ESTADO[secuencia[dia + 1]]   # estado siguiente → fila
        conteo[i][j] += 1

    matriz = np.zeros((n, n))
    for j in range(n):
        total = conteo[:, j].sum()
        if total > 0:
            matriz[:, j] = conteo[:, j] / total

    return matriz, conteo


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
st.caption("Estados: N, PN, PS, S — con matriz estimada desde datos simulados.")

# ============================================================
# INTERFAZ (STREAMLIT)
# - Reemplaza los input()/print() de consola por controles en la barra lateral.
# ============================================================

with st.sidebar:
    st.header("Parámetros")
    n_dias = st.slider("Número de días", min_value=60, max_value=90, value=75, step=1)
    usar_semilla = st.checkbox("Usar semilla fija (reproducible)", value=True)
    # En vez de usar un lambda, definimos una función normal para el texto.
    def _formatear_estado(codigo: str) -> str:
        return f"{codigo} — {NOMBRES_ESTADOS[codigo]}"

    estado_dia_n = st.selectbox(
        "Estado del día n (para predicción)",
        options=ESTADOS,
        format_func=_formatear_estado,
        index=0,
    )
    pasos_pred = st.slider("Días a predecir", min_value=1, max_value=10, value=4, step=1)

seed_value = 2026 if usar_semilla else None

colA, colB, colC, colD = st.columns(4)
colA.metric("Días", n_dias)
colB.metric("Estado día n", f"{estado_dia_n} ({NOMBRES_ESTADOS[estado_dia_n]})")
colC.metric("Semilla", "Fija" if usar_semilla else "Aleatoria")
colD.metric("Estados", ", ".join(ESTADOS))

st.divider()

if st.button("Generar simulación", type="primary"):
    secuencia = generar_secuencia_climatica(n_dias=n_dias, usar_semilla=usar_semilla)
    st.session_state["secuencia"] = secuencia

secuencia = st.session_state.get("secuencia")
if secuencia is None:
    st.info("Configura los parámetros y haz clic en **Generar simulación**.")
    st.stop()

df_secuencia = build_results_df(secuencia)
freq_df = build_frequency_df(secuencia)
A_df, conteo_df = construir_matriz_transicion_df(secuencia)

st.subheader("Vista por días")
colP, colU = st.columns(2)
colP.write(f"Primeros 15 días: {secuencia[:15]}")
colU.write(f"Últimos 15 días: {secuencia[-15:]}")

state_to_idx = {s: i for i, s in enumerate(ESTADOS)}
y_vals = [state_to_idx[s] for s in secuencia]
x_vals = list(range(1, len(secuencia) + 1))

fig_timeline, ax_t = plt.subplots(figsize=(12, 3.5))
ax_t.plot(x_vals, y_vals, color="#9E9E9E", linewidth=1, alpha=0.45)

for code in ESTADOS:
    xs = [i + 1 for i, s in enumerate(secuencia) if s == code]
    ys = [state_to_idx[code]] * len(xs)
    ax_t.scatter(xs, ys, color=COLORES_ESTADOS[code], s=28, label=f"{code} — {NOMBRES_ESTADOS[code]}")

ax_t.set_xlabel("Día")
ax_t.set_ylabel("Estado")
ax_t.set_yticks(list(range(len(ESTADOS))))
ax_t.set_yticklabels([f"{s} — {NOMBRES_ESTADOS[s]}" for s in ESTADOS])
ax_t.grid(axis="x", alpha=0.1)
ax_t.grid(axis="y", alpha=0.2)
ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
plt.tight_layout()
st.pyplot(fig_timeline, use_container_width=True)

left, right = st.columns([1.25, 1])

with left:
    st.subheader("Tabla de resultados")
    st.dataframe(df_secuencia, use_container_width=True, height=520)

with right:
    st.subheader("Distribución de frecuencias")

    fig_bar, ax_bar = plt.subplots(figsize=(8, 4.5))
    nombres = [NOMBRES_ESTADOS[e] for e in ESTADOS]
    valores = [int(freq_df.loc[freq_df["Código"] == e, "Días"].iloc[0]) for e in ESTADOS]
    colores = [COLORES_ESTADOS[e] for e in ESTADOS]
    barras = ax_bar.bar(nombres, valores, color=colores, edgecolor="white", linewidth=1.2)

    for barra, val in zip(barras, valores):
        pct = (val / max(1, n_dias)) * 100
        ax_bar.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 0.3,
            f"{val}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax_bar.set_ylabel("Número de días", fontsize=10)
    ax_bar.set_title(
        f"Distribución de estados climáticos — {n_dias} días simulados",
        fontsize=11,
        fontweight="bold",
    )
    ax_bar.set_ylim(0, max(valores) * 1.25 if max(valores) > 0 else 1)
    ax_bar.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    freq_mostrar = freq_df[["Código", "Estado", "Días", "Porcentaje"]].copy()
    porcentajes_texto = []
    for v in freq_mostrar["Porcentaje"].to_list():
        porcentajes_texto.append(f"{float(v):.2f}%")
    freq_mostrar["Porcentaje"] = porcentajes_texto

    st.dataframe(freq_mostrar, use_container_width=True, hide_index=True)

st.divider()

with st.expander("Ver matriz de transición estimada (usada en la simulación)", expanded=False):
    st.caption("Columnas = estado actual, filas = siguiente estado. Cada columna suma 1.")
    st.dataframe(A_df.style.format("{:.3f}"), use_container_width=True)
    col_sums = A_df.sum(axis=0)
    st.dataframe(pd.DataFrame({"Suma de columna": col_sums}).T.style.format("{:.3f}"), use_container_width=True)

    fig_m, ax_m = plt.subplots(figsize=(7, 5))
    im = ax_m.imshow(A_df.to_numpy(dtype=float), cmap="Purples", vmin=0, vmax=1)
    fig_m.colorbar(im, ax=ax_m, label="Probabilidad de transición")

    ax_m.set_xticks(range(len(ESTADOS)))
    ax_m.set_yticks(range(len(ESTADOS)))
    ax_m.set_xticklabels([f"Desde\n{e}" for e in ESTADOS], fontsize=10)
    ax_m.set_yticklabels([f"Hacia {e}" for e in ESTADOS], fontsize=10)

    A_np = A_df.to_numpy(dtype=float)
    for i in range(len(ESTADOS)):
        for j in range(len(ESTADOS)):
            color_texto = "white" if A_np[i, j] > 0.45 else "black"
            ax_m.text(
                j,
                i,
                f"{A_np[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color_texto,
            )

    ax_m.set_title("Matriz de Transición — Cadena de Márkov Climática", fontsize=11, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_m, use_container_width=True)

with st.expander("Ver conteo de transiciones (observado)", expanded=False):
    st.dataframe(conteo_df.astype(int), use_container_width=True)

st.subheader("Predicción de estados futuros")
predicciones = predecir_estados_futuros(A_df.to_numpy(dtype=float), estado_dia_n, pasos=int(pasos_pred))

columnas = []
datos = {}
for key in predicciones:
    columnas.append(key)
    datos[key] = predicciones[key]

index = []
for e in ESTADOS:
    index.append(f"{e} — {NOMBRES_ESTADOS[e]}")

pred_df = pd.DataFrame(datos, index=index)
st.dataframe(pred_df.style.format("{:.3f}"), use_container_width=True)

most_prob = []
for p in range(1, int(pasos_pred) + 1):
    vec = predicciones[f"n+{p}"]
    idx_max = int(np.argmax(vec))
    estado_probable = ESTADOS[idx_max]
    prob = float(vec[idx_max])
    fila = {
        "Día": f"n+{p}",
        "Más probable": f"{estado_probable} — {NOMBRES_ESTADOS[estado_probable]}",
        "Probabilidad": prob,
    }
    most_prob.append(fila)

df_most = pd.DataFrame(most_prob)

prob_texto = []
for x in df_most["Probabilidad"].to_list():
    prob_texto.append(f"{float(x) * 100:.2f}%")
df_most["Probabilidad"] = prob_texto

st.dataframe(df_most, use_container_width=True, hide_index=True)

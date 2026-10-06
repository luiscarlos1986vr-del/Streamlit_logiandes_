from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------
# Configuración global y constantes
# ---------------------------------------------------------------------
st.set_page_config(page_title="LogiAndes - Panel BI", layout="wide")

MESES = {1: "Enero", 2: "Febrero", 3: "Marzo"}
ORDEN_MES = list(MESES.values())

METAS = {
    "tasa_reclamos": {"meta": 30.0, "umbral": 45.0},
    "mediana_entrega": {"meta": 130.0, "umbral": 150.0},
    "pct_tarde": {"meta": 2.0, "umbral": 5.0},
    "ventas_riesgo": {"meta": 3.0, "umbral": 5.0},
    "concentracion_top2": {"meta": 60.0, "umbral": 65.0},
}

UMBRAL_TARDE_MIN = 240   # Definimos "entrega tardía" como más de 4 horas
MIN_PEDIDOS = 100        # Muestra mínima para advertir inestabilidad
ORDEN_CANAL = ["Estándar urbano", "Express", "Programado"]
COLOR_CANAL = ["#2a7f8f", "#d1603d", "#8c8c8c"]
COLOR_REGION = {"Costa": "#d1603d", "Sierra": "#2a7f8f"}
ICONO = {"Verde": "🟢", "Amarillo": "🟡", "Rojo": "🔴"}
FUENTE = "Fuente: LogiAndes S.A., logiandes.csv (enero-marzo 2026)"

URL = ("https://docs.google.com/spreadsheets/d/1z68Fjbvpb-SiFSJK_fnaN4dmljm0Yv-yZ3gunGizvb8/"
       "export?format=csv&gid=45425025")
LOCAL = Path(__file__).parent / "logiandes.csv"

# ---------------------------------------------------------------------
# Funciones auxiliares de datos y lógica de negocio
# ---------------------------------------------------------------------
def preparar(df):
    df = df.copy()
    df["fecha_pedido"] = pd.to_datetime(df["fecha_pedido"], errors="coerce")
    df["mes"] = pd.Categorical(df["fecha_pedido"].dt.month.map(MESES),
                               categories=ORDEN_MES, ordered=True)
    df["tarde"] = (df["tiempo_individual_entrega_min"] > UMBRAL_TARDE_MIN).astype(int)
    df["ventas_riesgo"] = df["ventas_asociadas_usd"] * df["reclamos_registrados"]
    return df


def wilson(k, n, z=1.96):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = k / n
    den = 1 + z ** 2 / n
    centro = (p + z ** 2 / (2 * n)) / den
    margen = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (centro - margen) * 1000, (centro + margen) * 1000


def agrupar(df, claves):
    g = (df.groupby(claves, observed=True)
           .agg(pedidos=("pedido_id", "count"),
                reclamos=("reclamos_registrados", "sum"),
                ventas=("ventas_asociadas_usd", "sum"),
                ventas_riesgo=("ventas_riesgo", "sum"),
                mediana_entrega=("tiempo_individual_entrega_min", "median"),
                pct_tarde=("tarde", "mean"))
           .reset_index())
    g["tasa_reclamos"] = g["reclamos"] / g["pedidos"] * 1000
    g["pct_tarde"] = g["pct_tarde"] * 100
    g["ic_bajo"], g["ic_alto"] = wilson(g["reclamos"], g["pedidos"])
    return g


def calcular_kpis(df):
    ventas_prov = df.groupby("provincia")["ventas_asociadas_usd"].sum().sort_values(ascending=False)
    total = df["ventas_asociadas_usd"].sum()
    return {
        "tasa_reclamos": df["reclamos_registrados"].mean() * 1000,
        "mediana_entrega": df["tiempo_individual_entrega_min"].median(),
        "pct_tarde": df["tarde"].mean() * 100,
        "ventas_riesgo": df["ventas_riesgo"].sum() / total * 100,
        "concentracion_top2": ventas_prov.head(2).sum() / total * 100,
    }


def estado(clave, valor):
    m = METAS[clave]
    if valor <= m["meta"]:
        return "Verde"
    if valor <= m["umbral"]:
        return "Amarillo"
    return "Rojo"


def formatear(clave, valor):
    if clave == "tasa_reclamos":
        return f"{valor:.1f}"
    if clave == "mediana_entrega":
        return f"{valor:.0f} min"
    return f"{valor:.1f}%"


# Fichas técnicas de los KPIs
FICHAS = [
    {
        "clave": "tasa_reclamos",
        "nombre": "Tasa de reclamos por 1.000 pedidos",
        "objetivo": "Medir la calidad percibida del servicio sin que el volumen distorsione la comparación.",
        "formula": "(Σ reclamos_registrados / nº de pedidos) × 1.000",
        "fuente": "logiandes.csv: reclamos_registrados, pedido_id",
        "frecuencia": "Mensual",
        "unidad": "reclamos por 1.000 pedidos",
        "interpretacion": "Menor es mejor. En subgrupos pequeños puede cambiar mucho con 1 o 2 casos, por eso mostramos el intervalo de confianza.",
        "audiencia": "Atención al cliente, gerencia general",
    },
    {
        "clave": "mediana_entrega",
        "nombre": "Mediana del tiempo de entrega",
        "objetivo": "Seguir la rapidez típica de la entrega sin que los valores extremos distorsionen el resultado.",
        "formula": "mediana(tiempo_individual_entrega_min)",
        "fuente": "logiandes.csv: tiempo_individual_entrega_min",
        "frecuencia": "Semanal",
        "unidad": "minutos",
        "interpretacion": "Menor es mejor. Depende del canal (Programado es lento a propósito) y de la distancia, así que se compara dentro de un mismo canal.",
        "audiencia": "Operaciones",
    },
    {
        "clave": "pct_tarde",
        "nombre": "% de entregas tardías (más de 240 min)",
        "objetivo": "Detectar la cola de entregas muy lentas que la mediana no muestra.",
        "formula": "(pedidos con tiempo_individual_entrega_min > 240 / nº de pedidos) × 100",
        "fuente": "logiandes.csv: tiempo_individual_entrega_min",
        "frecuencia": "Semanal",
        "unidad": "% de pedidos",
        "interpretacion": "Menor es mejor. El umbral de 240 min es una definición nuestra y Operaciones debería validarla.",
        "audiencia": "Operaciones, atención al cliente",
    },
    {
        "clave": "ventas_riesgo",
        "nombre": "% de ventas en riesgo por reclamo",
        "objetivo": "Dar a Finanzas una medida del dinero asociado a pedidos con reclamo, sin inventar rentabilidad.",
        "formula": "(Σ ventas_asociadas_usd de pedidos con reclamo / Σ ventas_asociadas_usd) × 100",
        "fuente": "logiandes.csv: ventas_asociadas_usd, reclamos_registrados",
        "frecuencia": "Mensual",
        "unidad": "% de las ventas",
        "interpretacion": "Menor es mejor. No es una pérdida real: es la venta de pedidos con reclamo. La base no trae costos ni devoluciones.",
        "audiencia": "Finanzas, gerencia general",
    },
    {
        "clave": "concentracion_top2",
        "nombre": "Concentración de ventas en las 2 provincias principales",
        "objetivo": "Medir cuánto depende el negocio de dos provincias.",
        "formula": "(ventas de las 2 provincias con más ventas / ventas totales) × 100",
        "fuente": "logiandes.csv: provincia, ventas_asociadas_usd",
        "frecuencia": "Trimestral",
        "unidad": "% de las ventas",
        "interpretacion": "Menor indica menos dependencia. Con una sola provincia filtrada vale 100% y no tiene sentido.",
        "audiencia": "Área comercial, gerencia general",
    },
]

for _f in FICHAS:
    _f["meta"] = f"≤ {METAS[_f['clave']]['meta']:g} {_f['unidad']}"
    _f["umbral"] = f"> {METAS[_f['clave']]['umbral']:g} {_f['unidad']} (rojo)"


# ---------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------
@st.cache_data
def cargar():
    df = pd.read_csv(LOCAL if LOCAL.exists() else URL)
    return preparar(df)


df = cargar()

# ---------------------------------------------------------------------
# Filtros globales en barra lateral
# ---------------------------------------------------------------------
st.sidebar.header("Filtros")
meses = st.sidebar.multiselect("Mes", ORDEN_MES, default=ORDEN_MES)
provincias_all = sorted(df["provincia"].unique())
provincias = st.sidebar.multiselect("Provincia", provincias_all, default=provincias_all)
canales = st.sidebar.multiselect("Canal de entrega", ORDEN_CANAL, default=ORDEN_CANAL)
tipos_all = sorted(df["tipo_cliente"].dropna().unique())
tipos = st.sidebar.multiselect("Tipo de cliente", tipos_all, default=tipos_all)
st.sidebar.caption("Los filtros afectan a todas las secciones. Las metas de los KPIs son "
                   "supuestos del grupo (ver pestaña Documentación).")

f = df[df["mes"].isin(meses) & df["provincia"].isin(provincias)
       & df["canal_entrega"].isin(canales) & df["tipo_cliente"].isin(tipos)]

st.title("LogiAndes: panel de servicio y ventas")
st.caption(FUENTE)

if f.empty:
    st.warning("No hay pedidos con los filtros seleccionados.")
    st.stop()
if len(f) < MIN_PEDIDOS:
    st.info(f"Los filtros dejan solo {len(f)} pedidos. Los indicadores pueden ser poco estables.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Vista ejecutiva", "Análisis territorial", "Detalle operativo", "Documentación"])

# =====================================================================
# 1. Vista ejecutiva
# =====================================================================
with tab1:
    valores = calcular_kpis(f)
    st.subheader("¿Cómo estamos frente a las metas?")
    cols = st.columns(5)
    for col, ficha in zip(cols, FICHAS):
        clave = ficha["clave"]
        est = estado(clave, valores[clave])
        col.metric(f"{ICONO[est]} {ficha['nombre']}", formatear(clave, valores[clave]))
        col.caption(f"Meta {ficha['meta']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Ventas (USD)", f"${f['ventas_asociadas_usd'].sum():,.0f}")
    c2.metric("Pedidos", f"{len(f):,}")
    c3.metric("Venta promedio por pedido (USD)", f"${f['ventas_asociadas_usd'].mean():,.2f}")

    mensual = agrupar(f, ["mes"])
    fig_m = make_subplots(specs=[[{"secondary_y": True}]])
    fig_m.add_bar(x=mensual["mes"], y=mensual["ventas"], name="Ventas (USD)",
                  marker_color="#9db8d3")
    fig_m.add_scatter(x=mensual["mes"], y=mensual["tasa_reclamos"], name="Reclamos por 1.000",
                      mode="lines+markers", line_color="#d1603d", secondary_y=True)
    fig_m.update_yaxes(title_text="Ventas (USD)", rangemode="tozero", secondary_y=False)
    fig_m.update_yaxes(title_text="Reclamos por 1.000 pedidos", rangemode="tozero", secondary_y=True)
    fig_m.update_layout(title="Ventas y reclamos por mes", template="plotly_white",
                        legend=dict(orientation="h", y=-0.2))

    prov = agrupar(f, ["provincia", "region"])
    tasa_global = valores["tasa_reclamos"]
    fig_b = px.scatter(prov, x="ventas", y="tasa_reclamos", size="pedidos", color="region",
                       text="provincia", color_discrete_map=COLOR_REGION,
                       labels={"ventas": "Ventas (USD)", "tasa_reclamos": "Reclamos por 1.000 pedidos",
                               "region": "Región"},
                       title="Peso de cada provincia en ventas frente a su tasa de reclamos",
                       template="plotly_white", size_max=45)
    fig_b.update_traces(textposition="top center")
    fig_b.add_hline(y=tasa_global, line_dash="dash", line_color="gray",
                    annotation_text=f"Promedio ({tasa_global:.1f})")
    fig_b.update_xaxes(rangemode="tozero")
    fig_b.update_yaxes(rangemode="tozero")

    g1, g2 = st.columns(2)
    g1.plotly_chart(fig_m, use_container_width=True)
    g2.plotly_chart(fig_b, use_container_width=True)
    st.caption("Tamaño del círculo: número de pedidos. " + FUENTE)

    peor = prov.sort_values("tasa_reclamos").iloc[-1]
    st.info(f"Lectura rápida: con los filtros actuales, {peor['provincia']} tiene la tasa de "
            f"reclamos más alta ({peor['tasa_reclamos']:.1f} por 1.000, sobre {int(peor['pedidos']):,} "
            f"pedidos). Es una asociación, no prueba la causa.")

# =====================================================================
# 2. Análisis territorial
# =====================================================================
with tab2:
    st.subheader("Comparación entre provincias")
    opciones = {
        "Reclamos por 1.000 pedidos": "tasa_reclamos",
        "Mediana de entrega (min)": "mediana_entrega",
        "% de entregas tardías": "pct_tarde",
        "Ventas (USD)": "ventas",
        "Ventas en riesgo (USD)": "ventas_riesgo",
    }
    etiqueta = st.selectbox("Indicador a comparar", list(opciones))
    col_m = opciones[etiqueta]

    t = agrupar(f, ["provincia", "region"]).sort_values(col_m)
    t["err_alto"] = t["ic_alto"] - t["tasa_reclamos"]
    t["err_bajo"] = t["tasa_reclamos"] - t["ic_bajo"]
    con_ic = col_m == "tasa_reclamos"
    fig_t = px.bar(t, x=col_m, y="provincia", orientation="h", color="region",
                   color_discrete_map=COLOR_REGION, text=col_m,
                   error_x="err_alto" if con_ic else None,
                   error_x_minus="err_bajo" if con_ic else None,
                   custom_data=["pedidos", "reclamos"],
                   labels={col_m: etiqueta, "provincia": "", "region": "Región"},
                   title=f"{etiqueta} por provincia", template="plotly_white")
    fig_t.update_traces(texttemplate="%{x:,.1f}", textposition="outside", cliponaxis=False,
                        hovertemplate="<b>%{y}</b><br>" + etiqueta + ": %{x:,.1f}"
                                      "<br>Pedidos: %{customdata[0]:,}<br>Reclamos: %{customdata[1]}"
                                      "<extra></extra>")
    if con_ic:
        st.caption("Las barras de error son el intervalo de confianza al 95% de la tasa.")
    tope = (t["ic_alto"].max() if con_ic else t[col_m].max()) * 1.3
    fig_t.update_layout(xaxis_range=[0, tope])
    st.plotly_chart(fig_t, use_container_width=True)

    st.subheader("Provincia por canal")
    min_n = st.slider("Mínimo de pedidos para mostrar una celda", 0, 300, MIN_PEDIDOS, step=10,
                      help="Las celdas con menos pedidos quedan en blanco porque su tasa no es confiable.")
    pc = agrupar(f, ["provincia", "canal_entrega"])
    pc_ok = pc[pc["pedidos"] >= min_n]
    if pc_ok.empty:
        st.warning("Ninguna combinación cumple ese mínimo de pedidos.")
    else:
        z = pc_ok.pivot(index="provincia", columns="canal_entrega", values="tasa_reclamos")
        n = pc.pivot(index="provincia", columns="canal_entrega", values="pedidos")
        z = z.reindex(index=sorted(f["provincia"].unique()),
                      columns=[c for c in ORDEN_CANAL if c in pc["canal_entrega"].unique()])
        n = n.reindex(index=z.index, columns=z.columns)
        texto = z.round(1).astype(str) + "<br>(n=" + n.fillna(0).astype(int).astype(str) + ")"
        texto = texto.where(z.notna(), "")
        fig_h = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, text=texto.values,
                                     texttemplate="%{text}", colorscale="Oranges",
                                     colorbar_title="Por 1.000", hoverongaps=False))
        fig_h.update_layout(title="Reclamos por 1.000 pedidos (n = pedidos de la celda)",
                            template="plotly_white", yaxis_autorange="reversed")
        st.plotly_chart(fig_h, use_container_width=True)
    st.caption(FUENTE)

# =====================================================================
# 3. Detalle operativo
# =====================================================================
with tab3:
    st.subheader("Tiempos, distancia y reclamos")
    o1, o2 = st.columns(2)

    with o1:
        muestra = f.sample(min(len(f), 2000), random_state=1)
        fig_s = px.scatter(muestra, x="distancia_km", y="tiempo_individual_entrega_min",
                           color="canal_entrega", opacity=0.5,
                           category_orders={"canal_entrega": ORDEN_CANAL},
                           color_discrete_sequence=COLOR_CANAL,
                           labels={"distancia_km": "Distancia (km)",
                                   "tiempo_individual_entrega_min": "Tiempo de entrega (min)",
                                   "canal_entrega": "Canal"},
                           title="Distancia y tiempo de entrega", template="plotly_white")
        fig_s.add_hline(y=UMBRAL_TARDE_MIN, line_dash="dash", line_color="gray",
                        annotation_text="240 min")
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("Mostramos una muestra de hasta 2.000 pedidos para que el gráfico responda rápido.")

    with o2:
        fig_c = px.box(f, x="canal_entrega", y="tiempo_individual_entrega_min",
                       color="canal_entrega", category_orders={"canal_entrega": ORDEN_CANAL},
                       color_discrete_sequence=COLOR_CANAL,
                       labels={"canal_entrega": "", "tiempo_individual_entrega_min": "Minutos"},
                       title="Tiempo de entrega por canal", template="plotly_white")
        fig_c.update_layout(showlegend=False)
        st.plotly_chart(fig_c, use_container_width=True)

    o3, o4 = st.columns(2)
    with o3:
        rec = f[f["reclamos_registrados"] == 1]
        if rec.empty:
            st.info("No hay reclamos con los filtros actuales.")
        else:
            motivos = (rec["motivo_reclamo"].fillna("Sin motivo").value_counts()
                          .rename_axis("motivo").reset_index(name="reclamos")
                          .sort_values("reclamos"))
            fig_mo = px.bar(motivos, x="reclamos", y="motivo", orientation="h", text="reclamos",
                            color_discrete_sequence=["#2a7f8f"],
                            labels={"reclamos": "Reclamos", "motivo": ""},
                            title="Reclamos por motivo", template="plotly_white")
            fig_mo.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig_mo, use_container_width=True)
    with o4:
        resumen_canal = agrupar(f, ["canal_entrega"]).set_index("canal_entrega")
        resumen_canal = resumen_canal.reindex([c for c in ORDEN_CANAL if c in resumen_canal.index])
        st.markdown("**Resumen por canal**")
        st.dataframe(resumen_canal[["pedidos", "reclamos", "tasa_reclamos", "mediana_entrega",
                                    "pct_tarde"]].round(1), use_container_width=True)

    st.subheader("Pedidos")
    solo = st.radio("Mostrar", ["Todos", "Solo con reclamo", "Solo entregas tardías"], horizontal=True)
    tabla = f
    if solo == "Solo con reclamo":
        tabla = f[f["reclamos_registrados"] == 1]
    elif solo == "Solo entregas tardías":
        tabla = f[f["tarde"] == 1]
    cols_tabla = ["pedido_id", "fecha_pedido", "provincia", "canal_entrega", "tipo_cliente",
                  "distancia_km", "tiempo_individual_entrega_min", "reclamos_registrados",
                  "motivo_reclamo", "ventas_asociadas_usd"]
    st.write(f"{len(tabla):,} pedidos")
    st.dataframe(tabla[cols_tabla].head(500), use_container_width=True, hide_index=True)
    st.download_button("Descargar pedidos filtrados (CSV)",
                       tabla[cols_tabla].to_csv(index=False).encode("utf-8"),
                       file_name="pedidos_filtrados.csv", mime="text/csv")
    st.caption("La tabla muestra los primeros 500 pedidos; la descarga trae todos los filtrados.")

# =====================================================================
# 4. Documentación
# =====================================================================
with tab4:
    st.subheader("Fichas técnicas de los KPIs")
    ficha_df = pd.DataFrame(FICHAS).drop(columns=["clave", "unidad"])
    ficha_df.columns = ["KPI", "Objetivo", "Fórmula", "Fuente", "Frecuencia", "Interpretación",
                        "Audiencia", "Meta", "Umbral"]
    ficha_df = ficha_df[["KPI", "Objetivo", "Fórmula", "Fuente", "Frecuencia", "Meta", "Umbral",
                         "Interpretación", "Audiencia"]]
    st.dataframe(ficha_df, use_container_width=True, hide_index=True)
    st.caption("Semáforo: 🟢 cumple la meta, 🟡 entre la meta y el umbral, 🔴 pasa el umbral.")

    st.subheader("Supuestos")
    st.markdown(
        "- La base no trae metas: **nosotros** las propusimos a partir del primer trimestre y "
        "deben validarse con cada área.\n"
        "- No usamos rentabilidad porque la base no tiene costos. «Ventas en riesgo» es la venta "
        "de pedidos con reclamo, no una pérdida real.\n"
        "- No limpiamos la base; solo tipamos la fecha y creamos columnas auxiliares.\n"
        "- Con tres meses de datos no podemos hablar de tendencias.")

    st.subheader("Riesgos y límites de lectura")
    st.markdown(
        "- **Sobreinteractividad:** muchos filtros permiten llegar a subgrupos con pocos pedidos. "
        "Por eso avisamos cuando hay menos de 100 pedidos y ocultamos celdas con muestra pequeña.\n"
        "- **Sesgos visuales:** las barras parten de cero, los ejes se mantienen fijos entre "
        "provincias y la tasa se muestra por 1.000 pedidos para no premiar a las provincias grandes.\n"
        "- **Privacidad:** la base no tiene datos personales del cliente. Si se agregaran, "
        "habría que anonimizarlos antes de mostrarlos en la tabla de detalle.\n"
        "- **Gobernanza:** fuente única, fórmulas documentadas y metas con responsable.\n"
        "- **Límites interpretativos:** el análisis es exploratorio. Dos variables que se mueven "
        "juntas no demuestran causa, y no controlamos distancia, canal y cliente a la vez.")
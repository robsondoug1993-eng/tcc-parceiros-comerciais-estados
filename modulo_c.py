"""
Módulo C - Análises avançadas (machine learning)
TCC: Principais parceiros comerciais dos estados brasileiros (Comex Stat 2022-2026)

1) PREVISÃO das exportações e importações mensais do Brasil (séries temporais)
   - Compara 3 modelos num período de teste (jan-set/2026, treino 2022-2025):
       a) Sazonal ingênuo (mesmo mês do ano anterior) - baseline
       b) Regressão linear com tendência + sazonalidade mensal (scikit-learn)
       c) Holt-Winters aditivo (implementado em numpy, parâmetros por grid search)
   - Escolhe o de menor MAPE, reajusta com todos os dados e prevê out/2026-dez/2027.

2) CLUSTERIZAÇÃO das UFs pelo perfil de comércio exterior (K-Means)
   - Indicadores de perfil: concentração de parceiros (HHI), % China,
     % EUA, % União Europeia, % Mercosul, concentração de produtos (HHI SH4),
     % produtos básicos (cap. 01-27) e preço médio (US$/kg, log).
     Crescimento 2022->2025, peso do 1º destino e equilíbrio da balança: descritivos.

   - k escolhido pela maior silhueta entre 3 e 5 grupos (mínimo 3 UFs por grupo).

Uso: python modulo_c.py <pasta_saida_do_ETL> <pasta_resultados>
Requer: pandas, numpy, scikit-learn, matplotlib (todos incluídos no Anaconda)
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

SAIDA = sys.argv[1] if len(sys.argv) > 1 else "saida"
RES = sys.argv[2] if len(sys.argv) > 2 else "resultados_modulo_c"
os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
AZUL, AZUL_ESC, LARANJA, CINZA = "#1f77d4", "#1a2a6c", "#e8702a", "#9aa0a6"

# ---------------------------------------------------------------- DADOS
fato = pd.read_csv(os.path.join(SAIDA, "fato_comercio.csv"), sep=";",
                   dtype={"CO_PAIS": str, "CO_SH4": str})
pais = pd.read_csv(os.path.join(SAIDA, "dim_pais.csv"), sep=";", dtype=str)
# mesmos filtros do dashboard: exclui UF EX/ND e códigos administrativos 100/105
fato = fato[~fato.SG_UF.isin(["EX", "ND"]) & ~fato.CO_PAIS.isin(["100", "105"])].copy()
fato["DATA"] = pd.to_datetime(fato.DATA)
fato["ANO"] = fato.DATA.dt.year

# ================================================================ 1) PREVISÃO
def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def holt_winters(y, alpha, beta, gamma, m=12, h=0):
    """Holt-Winters aditivo. Retorna (ajuste 1 passo à frente, previsão h passos)."""
    y = np.asarray(y, float)
    nivel = y[:m].mean()
    tend = (y[m:2 * m].mean() - y[:m].mean()) / m
    saz = list(y[:m] - nivel)
    ajuste = []
    for t in range(len(y)):
        s = saz[t % m] if t < m else saz[t - m]
        ajuste.append(nivel + tend + s)
        nivel_ant = nivel
        nivel = alpha * (y[t] - s) + (1 - alpha) * (nivel + tend)
        tend = beta * (nivel - nivel_ant) + (1 - beta) * tend
        saz.append(gamma * (y[t] - nivel) + (1 - gamma) * s)
    prev = [nivel + (k + 1) * tend + saz[len(y) - m + (k % m)] for k in range(h)]
    return np.array(ajuste), np.array(prev)


def ajusta_hw(y):
    grade = np.arange(0.05, 1.0, 0.1)
    melhor = None
    for a, b, g in itertools.product(grade, [0.01, 0.05, 0.1, 0.2], grade):
        aj, _ = holt_winters(y, a, b, g)
        sse = np.sum((y[12:] - aj[12:]) ** 2)
        if melhor is None or sse < melhor[0]:
            melhor = (sse, a, b, g)
    return melhor[1:]


def x_regressao(datas, t0):
    t = ((datas.year - t0.year) * 12 + (datas.month - t0.month)).values.reshape(-1, 1)
    meses = pd.get_dummies(datas.month).reindex(columns=range(1, 13), fill_value=0).values[:, 1:]
    return np.hstack([t, meses])


serie = fato.groupby(["DATA", "FLUXO"]).VL_FOB.sum().unstack() / 1e9   # US$ bi
corte = pd.Timestamp("2026-01-01")
metricas, previsoes = [], []
futuro = pd.date_range(serie.index.max() + pd.offsets.MonthBegin(1), "2027-12-01", freq="MS")

for fluxo, nome in [("EXP", "Exportação"), ("IMP", "Importação")]:
    y = serie[fluxo]
    tr, te = y[y.index < corte], y[y.index >= corte]
    h = len(te)
    # a) sazonal ingênuo
    p_naive = tr.iloc[-12:].values[:h]
    # b) regressão tendência + sazonalidade
    reg = LinearRegression().fit(x_regressao(tr.index, y.index[0]), tr.values)
    p_reg = reg.predict(x_regressao(te.index, y.index[0]))
    # c) Holt-Winters
    a, b, g = ajusta_hw(tr.values)
    _, p_hw = holt_winters(tr.values, a, b, g, h=h)
    res = {"Sazonal ingênuo": p_naive, "Regressão tendência+sazonal": p_reg,
           "Holt-Winters aditivo": p_hw}
    for modelo, p in res.items():
        metricas.append({"Fluxo": nome, "Modelo": modelo, "MAPE_%": round(mape(te.values, p), 2),
                         "MAE_US$_bi": round(float(np.mean(np.abs(te.values - p))), 2)})
    melhor = min(res, key=lambda k: mape(te.values, res[k]))
    erro_sd = float(np.std(te.values - res[melhor], ddof=1))
    # reajuste com todos os dados e previsão
    if melhor.startswith("Regressão"):
        reg = LinearRegression().fit(x_regressao(y.index, y.index[0]), y.values)
        p_fut = reg.predict(x_regressao(futuro, y.index[0]))
    elif melhor.startswith("Holt"):
        a, b, g = ajusta_hw(y.values)
        _, p_fut = holt_winters(y.values, a, b, g, h=len(futuro))
    else:
        p_fut = np.tile(y.iloc[-12:].values, 3)[:len(futuro)]
    for d, v in y.items():
        previsoes.append({"DATA": d.date(), "FLUXO": fluxo, "TIPO": "Real", "VALOR_US$_bi": round(v, 3)})
    for d, v in zip(te.index, res[melhor]):
        previsoes.append({"DATA": d.date(), "FLUXO": fluxo, "TIPO": "Teste (previsto)",
                          "VALOR_US$_bi": round(float(v), 3)})
    for d, v in zip(futuro, p_fut):
        previsoes.append({"DATA": d.date(), "FLUXO": fluxo, "TIPO": "Previsão",
                          "VALOR_US$_bi": round(float(v), 3),
                          "LIM_INF": round(float(v) - 1.96 * erro_sd, 3),
                          "LIM_SUP": round(float(v) + 1.96 * erro_sd, 3)})
    print(f"{nome}: melhor modelo = {melhor}")

met = pd.DataFrame(metricas)
prev = pd.DataFrame(previsoes)
met.to_csv(os.path.join(RES, "previsao_metricas.csv"), sep=";", index=False, encoding="utf-8-sig")
prev.to_csv(os.path.join(RES, "previsao_series.csv"), sep=";", index=False, encoding="utf-8-sig")
print(met.to_string(index=False))

# resumo anual (real + previsto)
prev["ANO"] = pd.to_datetime(prev.DATA).dt.year
anual = (prev[prev.TIPO.isin(["Real", "Previsão"])]
         .groupby(["ANO", "FLUXO"])["VALOR_US$_bi"].sum().unstack().round(1))
anual["SALDO"] = (anual.EXP - anual.IMP).round(1)
anual.to_csv(os.path.join(RES, "previsao_anual.csv"), sep=";", encoding="utf-8-sig")
print(anual.to_string())

# gráfico
fig, ax = plt.subplots(figsize=(10, 4.2))
for fluxo, cor, nome in [("EXP", AZUL, "Exportação"), ("IMP", AZUL_ESC, "Importação")]:
    r = prev[(prev.FLUXO == fluxo) & (prev.TIPO == "Real")]
    p = prev[(prev.FLUXO == fluxo) & (prev.TIPO == "Previsão")]
    t = prev[(prev.FLUXO == fluxo) & (prev.TIPO == "Teste (previsto)")]
    ax.plot(pd.to_datetime(r.DATA), r["VALOR_US$_bi"], color=cor, lw=1.6, label=f"{nome} (real)")
    ax.plot(pd.to_datetime(t.DATA), t["VALOR_US$_bi"], color=cor, lw=1, ls=":", label=f"{nome} (teste)")
    ax.plot(pd.to_datetime(p.DATA), p["VALOR_US$_bi"], color=cor, lw=1.6, ls="--", label=f"{nome} (previsão)")
    ax.fill_between(pd.to_datetime(p.DATA), p.LIM_INF, p.LIM_SUP, color=cor, alpha=0.12)
ax.axvline(corte, color=CINZA, lw=0.8)
ax.axvline(futuro[0], color=CINZA, lw=0.8)
ax.set_title("Exportações e importações mensais do Brasil — real, teste e previsão até dez/2027")
ax.set_ylabel("US$ bilhões (FOB)")
ax.legend(ncol=3, fontsize=7, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(RES, "grafico_previsao.png"))
plt.close(fig)

# ================================================================ 2) CLUSTERIZAÇÃO
UE = set(pais.loc[pais.UNIAO_EUROPEIA == "Sim", "CO_PAIS"])
MERCOSUL = {"063", "586", "845"}          # Argentina, Paraguai, Uruguai
exp = fato[fato.FLUXO == "EXP"]
imp = fato[fato.FLUXO == "IMP"]


def hhi(s):
    p = s / s.sum()
    return float((p ** 2).sum())


linhas = []
for uf, g in exp.groupby("SG_UF"):
    tot = g.VL_FOB.sum()
    por_pais = g.groupby("CO_PAIS").VL_FOB.sum()
    por_prod = g.groupby("CO_SH4").VL_FOB.sum()
    por_ano = g.groupby("ANO").VL_FOB.sum()
    imp_uf = imp.loc[imp.SG_UF == uf, "VL_FOB"].sum()
    linhas.append({
        "UF": uf,
        "EXPORTACAO_US$_bi": tot / 1e9,
        "HHI_PAISES": hhi(por_pais),
        "PCT_1O_DESTINO": por_pais.max() / tot,
        "PCT_CHINA": por_pais.get("160", 0) / tot,
        "PCT_EUA": por_pais.get("249", 0) / tot,
        "PCT_UE": por_pais[por_pais.index.isin(UE)].sum() / tot,
        "PCT_MERCOSUL": por_pais[por_pais.index.isin(MERCOSUL)].sum() / tot,
        "HHI_PRODUTOS": hhi(por_prod),
        "PCT_BASICOS": g.loc[g.CO_SH4.str[:2].astype(int) <= 27, "VL_FOB"].sum() / tot,
        "LOG_PRECO_MEDIO": np.log(tot / g.KG_LIQUIDO.sum()),
        "CRESC_22_25": por_ano.get(2025, np.nan) / por_ano.get(2022, np.nan) - 1,
        "EQUILIBRIO_BALANCA": (tot - imp_uf) / (tot + imp_uf),
    })
perfil = pd.DataFrame(linhas).set_index("UF")
# Variáveis de PERFIL usadas no agrupamento. Crescimento e equilíbrio da balança ficam
# apenas descritivos: o crescimento tem outliers (ex.: RN, SE > +190%) que criavam grupos
# artificiais de 2 UFs.
features = ["HHI_PAISES", "PCT_CHINA", "PCT_EUA", "PCT_UE", "PCT_MERCOSUL",
            "HHI_PRODUTOS", "PCT_BASICOS", "LOG_PRECO_MEDIO"]
X = StandardScaler().fit_transform(perfil[features].fillna(0))

# Regra de escolha de k: maior silhueta entre 3 e 5 grupos, com no mínimo 3 UFs por grupo
silhuetas, k_best = {}, None
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
    silhuetas[k] = silhouette_score(X, km.labels_)
    tam_min = np.bincount(km.labels_).min()
    if 3 <= k <= 5 and tam_min >= 3 and (k_best is None or silhuetas[k] > silhuetas[k_best]):
        k_best = k
print("Silhueta por k:", {k: round(v, 3) for k, v in silhuetas.items()}, "-> k =", k_best)
km = KMeans(n_clusters=k_best, n_init=50, random_state=42).fit(X)
perfil["CLUSTER"] = km.labels_ + 1

# Nome automático de cada grupo a partir do centróide
cent = perfil.groupby("CLUSTER")[features].mean()
nomes = {}
c_china = cent.PCT_CHINA.idxmax()
nomes[c_china] = "Commodities concentradas na China"
resto = cent.drop(index=c_china)
c_norte = (resto.PCT_EUA + resto.PCT_UE).idxmax()
nomes[c_norte] = "Orientados a EUA e União Europeia"
for c in cent.index:
    nomes.setdefault(c, "Pauta e destinos diversificados")
perfil["GRUPO"] = perfil.CLUSTER.map(nomes)
features_desc = features + ["PCT_1O_DESTINO", "CRESC_22_25", "EQUILIBRIO_BALANCA"]
perfil.round(4).to_csv(os.path.join(RES, "clusters_uf.csv"), sep=";", encoding="utf-8-sig")
resumo = perfil.groupby("CLUSTER")[features_desc + ["EXPORTACAO_US$_bi"]].mean().round(3)
resumo.insert(0, "GRUPO", resumo.index.map(nomes))
resumo["N_UFs"] = perfil.groupby("CLUSTER").size()
resumo["EXPORTACAO_TOTAL_US$_bi"] = perfil.groupby("CLUSTER")["EXPORTACAO_US$_bi"].sum().round(1)
resumo["UFs"] = perfil.groupby("CLUSTER").apply(lambda d: ", ".join(sorted(d.index)))
resumo.to_csv(os.path.join(RES, "clusters_resumo.csv"), sep=";", encoding="utf-8-sig")
pd.Series(silhuetas, name="silhueta").to_csv(os.path.join(RES, "clusters_silhueta.csv"), sep=";")
print(resumo.T.to_string())

# gráfico PCA dos grupos
pcs = PCA(n_components=2, random_state=42).fit(X)
Z = pcs.transform(X)
cores = [AZUL, LARANJA, "#2ca58d", "#8e44ad", "#c0392b", CINZA]
fig, ax = plt.subplots(figsize=(7.5, 5))
for c in sorted(perfil.CLUSTER.unique()):
    msk = perfil.CLUSTER.values == c
    ax.scatter(Z[msk, 0], Z[msk, 1], s=40 + 6 * np.sqrt(perfil["EXPORTACAO_US$_bi"][msk]),
               color=cores[c - 1], alpha=0.85, label=f"Grupo {c}: {nomes[c]}")
for i, uf in enumerate(perfil.index):
    ax.annotate(uf, (Z[i, 0], Z[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ev = pcs.explained_variance_ratio_ * 100
ax.set_xlabel(f"Componente 1 ({ev[0]:.0f}% da variância)")
ax.set_ylabel(f"Componente 2 ({ev[1]:.0f}% da variância)")
ax.set_title(f"Agrupamento das UFs por perfil de exportação (K-Means, k={k_best})")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "grafico_clusters.png"))
plt.close(fig)
print("Resultados gravados em", RES)

"""
ETL - Principais parceiros comerciais dos estados brasileiros (Comex Stat / MDIC)

Entrada (pasta RAW): EXP_aaaa_MUN.csv, IMP_aaaa_MUN.csv, NCM_SH.csv, PAIS.csv, UF_MUN.csv
Saída  (pasta OUT): modelo estrela em CSV (UTF-8, separador ;) pronto para o Power BI
    fato_comercio.csv  - ano, mês, fluxo, UF, país, SH4 -> kg líquido, valor FOB (US$)
    dim_tempo.csv      - data (1º dia do mês), ano, mês, trimestre, semestre
    dim_uf.csv         - sigla, nome do estado, região
    dim_pais.csv       - código, nome, ISO3, flag União Europeia
    dim_produto.csv    - SH4 -> SH2 (capítulo) -> Seção
    dim_fluxo.csv      - Exportação / Importação
    validacao_totais.csv - totais por ano e fluxo, brutos vs. fato (homologação)

Uso: python etl_comex.py <pasta_raw> <pasta_saida>
"""
import sys
import glob
import os
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "./saida"
os.makedirs(OUT, exist_ok=True)

# Códigos de UF que não são estados (EX = exterior, ND = não declarada, etc.)
UF_INFO = {
    "AC": ("Acre", "Norte"), "AP": ("Amapá", "Norte"), "AM": ("Amazonas", "Norte"),
    "PA": ("Pará", "Norte"), "RO": ("Rondônia", "Norte"), "RR": ("Roraima", "Norte"),
    "TO": ("Tocantins", "Norte"),
    "AL": ("Alagoas", "Nordeste"), "BA": ("Bahia", "Nordeste"), "CE": ("Ceará", "Nordeste"),
    "MA": ("Maranhão", "Nordeste"), "PB": ("Paraíba", "Nordeste"), "PE": ("Pernambuco", "Nordeste"),
    "PI": ("Piauí", "Nordeste"), "RN": ("Rio Grande do Norte", "Nordeste"), "SE": ("Sergipe", "Nordeste"),
    "DF": ("Distrito Federal", "Centro-Oeste"), "GO": ("Goiás", "Centro-Oeste"),
    "MT": ("Mato Grosso", "Centro-Oeste"), "MS": ("Mato Grosso do Sul", "Centro-Oeste"),
    "ES": ("Espírito Santo", "Sudeste"), "MG": ("Minas Gerais", "Sudeste"),
    "RJ": ("Rio de Janeiro", "Sudeste"), "SP": ("São Paulo", "Sudeste"),
    "PR": ("Paraná", "Sul"), "RS": ("Rio Grande do Sul", "Sul"), "SC": ("Santa Catarina", "Sul"),
}

# Países da União Europeia (27) pelo ISO3
UE_ISO3 = {"AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA", "DEU", "GRC",
           "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK",
           "SVN", "ESP", "SWE"}

# ---------------------------------------------------------------- EXTRAÇÃO
DTYPES = {"CO_ANO": "int16", "CO_MES": "int8", "SH4": str, "CO_PAIS": str,
          "SG_UF_MUN": str, "KG_LIQUIDO": "int64", "VL_FOB": "int64"}
COLS = list(DTYPES)

partes, validacao = [], []
for fluxo in ("EXP", "IMP"):
    for arq in sorted(glob.glob(os.path.join(RAW, f"{fluxo}_*_MUN.csv"))):
        df = pd.read_csv(arq, sep=";", usecols=COLS, dtype=DTYPES)
        df["FLUXO"] = fluxo
        print(f"lido {os.path.basename(arq)}: {len(df):,} linhas")
        validacao.append(df.groupby(["CO_ANO", "FLUXO"])[["VL_FOB", "KG_LIQUIDO"]].sum())
        # ---------------------------------------------------- TRANSFORMAÇÃO
        # Agrega o município -> UF (granularidade do projeto)
        partes.append(df.groupby(["CO_ANO", "CO_MES", "FLUXO", "SG_UF_MUN", "CO_PAIS", "SH4"],
                                 as_index=False)[["KG_LIQUIDO", "VL_FOB"]].sum())

fato = pd.concat(partes, ignore_index=True)
fato["DATA"] = pd.to_datetime(dict(year=fato.CO_ANO, month=fato.CO_MES, day=1)).dt.date
fato = fato.rename(columns={"SG_UF_MUN": "SG_UF", "SH4": "CO_SH4"})
fato = fato[["DATA", "FLUXO", "SG_UF", "CO_PAIS", "CO_SH4", "KG_LIQUIDO", "VL_FOB"]]

# ---------------------------------------------------------------- DIMENSÕES
datas = pd.to_datetime(sorted(fato.DATA.unique()))
dim_tempo = pd.DataFrame({"DATA": datas.date, "ANO": datas.year, "MES": datas.month,
                          "NOME_MES": datas.month.map({1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai",
                                                       6: "Jun", 7: "Jul", 8: "Ago", 9: "Set",
                                                       10: "Out", 11: "Nov", 12: "Dez"}),
                          "TRIMESTRE": "T" + datas.quarter.astype(str),
                          "SEMESTRE": "S" + ((datas.month - 1) // 6 + 1).astype(str)})

ufs = sorted(fato.SG_UF.unique())
CO_UF_IBGE = {"RO": 11, "AC": 12, "AM": 13, "RR": 14, "PA": 15, "AP": 16, "TO": 17, "MA": 21,
              "PI": 22, "CE": 23, "RN": 24, "PB": 25, "PE": 26, "AL": 27, "SE": 28, "BA": 29,
              "MG": 31, "ES": 32, "RJ": 33, "SP": 35, "PR": 41, "SC": 42, "RS": 43, "MS": 50,
              "MT": 51, "GO": 52, "DF": 53}
# Código IBGE numérico (99 = não declarada/exterior): também garante a detecção do cabeçalho
dim_uf = pd.DataFrame({"CO_UF_IBGE": [CO_UF_IBGE.get(u, 99) for u in ufs], "SG_UF": ufs,
                       "NO_UF": [UF_INFO.get(u, ("Não declarada / Exterior", ""))[0] for u in ufs],
                       "REGIAO": [UF_INFO.get(u, ("", "Outros"))[1] for u in ufs]})

pais = pd.read_csv(os.path.join(RAW, "PAIS.csv"), sep=";", dtype=str, encoding="latin1")
dim_pais = pais[["CO_PAIS", "NO_PAIS", "NO_PAIS_ING", "CO_PAIS_ISOA3"]].copy()
dim_pais["UNIAO_EUROPEIA"] = dim_pais.CO_PAIS_ISOA3.isin(UE_ISO3).map({True: "Sim", False: "Não"})
# Territórios que partilham ISO3 com países da UE (ex.: Madeira, Canárias) entram como UE também
dim_pais = dim_pais[dim_pais.CO_PAIS.isin(fato.CO_PAIS.unique())]

sh = pd.read_csv(os.path.join(RAW, "NCM_SH.csv"), sep=";", dtype=str, encoding="latin1")
dim_produto = (sh[["CO_SH4", "NO_SH4_POR", "CO_SH2", "NO_SH2_POR", "CO_NCM_SECROM", "NO_SEC_POR"]]
               .drop_duplicates("CO_SH4"))
faltantes = set(fato.CO_SH4.unique()) - set(dim_produto.CO_SH4)
if faltantes:  # SH4 sem descrição (raro) -> linha "Não classificado"
    dim_produto = pd.concat([dim_produto, pd.DataFrame({"CO_SH4": sorted(faltantes),
                             "NO_SH4_POR": "Não classificado", "CO_SH2": [c[:2] for c in sorted(faltantes)],
                             "NO_SH2_POR": "Não classificado", "CO_NCM_SECROM": "-",
                             "NO_SEC_POR": "Não classificado"})])
dim_produto = dim_produto[dim_produto.CO_SH4.isin(fato.CO_SH4.unique())]

# Coluna numérica ORDEM: garante que o Power BI reconheça a 1ª linha como cabeçalho
dim_fluxo = pd.DataFrame({"FLUXO": ["EXP", "IMP"], "NO_FLUXO": ["Exportação", "Importação"],
                          "ORDEM": [1, 2]})

# ---------------------------------------------------------------- VALIDAÇÃO
bruto = pd.concat(validacao).groupby(level=[0, 1]).sum()
fato_tot = (fato.assign(CO_ANO=pd.to_datetime(fato.DATA).dt.year)
            .groupby(["CO_ANO", "FLUXO"])[["VL_FOB", "KG_LIQUIDO"]].sum())
val = bruto.join(fato_tot, lsuffix="_BRUTO", rsuffix="_FATO").reset_index()
val["OK"] = (val.VL_FOB_BRUTO == val.VL_FOB_FATO) & (val.KG_LIQUIDO_BRUTO == val.KG_LIQUIDO_FATO)
assert val.OK.all(), "Totais da fato divergem dos arquivos brutos!"

# ---------------------------------------------------------------- CARGA
saidas = {"fato_comercio": fato, "dim_tempo": dim_tempo, "dim_uf": dim_uf, "dim_pais": dim_pais,
          "dim_produto": dim_produto, "dim_fluxo": dim_fluxo, "validacao_totais": val}
for nome, df in saidas.items():
    df.to_csv(os.path.join(OUT, f"{nome}.csv"), sep=";", index=False, encoding="utf-8-sig")
    print(f"gravado {nome}.csv: {len(df):,} linhas")
print(val.to_string(index=False))

import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.express as px
from scipy.stats import pearsonr, linregress
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

st.set_page_config(
    page_title="Burnout",
    layout="wide",
)

url = "https://opendata.cbs.nl/#/CBS/nl/navigatieScherm/thema?themaNr=82493"
st.title(":red[Burnout]")
sLong = """
    In 2024 is zo\'n 4%% van de werkende Nederlanders uitgevallen door burnout.
    Burnoutklachten zijn de afgelopen 10 jaar flink toegenomen in Nederland, en stijgen nog altijd.
    Wanneer een werknemer uitvalt door burnout, duurt dit bovendien gemiddeld 2 maanden, veruit het langste van alle soorten verzuim.
    In dit dashboard duiken we in de [statistieken van het CBS](%s) rondom burnout. Wat zijn de risicogroepen, 
    wat zijn de gevolgen, en welke factoren kunnen een burnout voorspellen?
    """% url

st.write(sLong)

@st.cache_data
def load_data_sectoren() -> pd.DataFrame:
    # Data
    data_url = "https://opendata.cbs.nl/ODataApi/odata/86009NED/TypedDataSet?$filter=Marges eq 'MW00000'"
    df = pd.DataFrame(requests.get(data_url).json()["value"])

    # Bedrijfskenmerken labels
    labels_url = "https://opendata.cbs.nl/ODataApi/odata/86009NED/Bedrijfskenmerken"

    # Klachten leesbaar
    klachten_url = "https://opendata.cbs.nl/ODataApi/odata/86009NED/DataProperties"

    # Voeg leesbare labels toe uit de metadata
    labels = {}
    for item in requests.get(labels_url).json()["value"]:
        key = item["Key"].strip()
        title = item["Title"].strip()
        labels[key] = title

    df["Bedrijfskenmerken"] = df["Bedrijfskenmerken"].str.strip()
    df["Bedrijfstak_label"] = df["Bedrijfskenmerken"].map(labels)
    df["Kenmerk_type"] = df["Bedrijfskenmerken"].apply(
        lambda x: "Vestigingsgrootte" if str(x).startswith("WN") else "Bedrijfstak"
    )

    # Maak een mapping van klachten naar echte titels
    klachten = {}
    for item in requests.get(klachten_url).json()["value"]:
        key = item["Key"].strip()
        title = item["Title"].strip()
        klachten[key] = title

    # Haal de indexering uit de titel
    df["label_clean"] = df["Bedrijfstak_label"].str.replace(r"^[A-Z][-A-Z]* ", "", regex=True)

    # # Voeg leesbare redenen toe
    # for col, label in klachten.items():
    #
    #     df[label] = pd.to_numeric(df[col], errors="coerce")

    # De waardes waren verspreid over twee kolommen
    df["BurnoutCombined"] = df["PsychischOverspannenheidBurnOut_26"].combine_first(df["PsychischOverspannenheidBurnOut_13"])

    # Jaar normaal
    df["Jaar"] = df["Perioden"].str[0:4]


    # Burnout-score: kans dat werknemer uitvalt door burnout. Delen door honderd voor percentage
    df["burnout_score"] = (
        pd.to_numeric(df["AandeelWerknemersDieHebbenVerzuimd_1"], errors="coerce") *
        pd.to_numeric(df["BurnoutCombined"], errors="coerce")
    ) / 100

    # Deze gebruiken we als text label in grafieken
    df["burnout_label"] = df["burnout_score"].map("{:.2f}%".format)

    return df, klachten

df, klachten = load_data_sectoren()



@st.cache_data
def load_verzuimduur():
    url = "https://opendata.cbs.nl/ODataApi/odata/86168NED/TypedDataSet?$filter=Marges eq 'MW00000'"
    df = pd.DataFrame(requests.get(url).json()["value"])
    labels_url = "https://opendata.cbs.nl/ODataApi/odata/86168NED/KenmerkenVerzuimgeval"

    # Voeg leesbare labels toe uit de metadata
    labels = {}
    for item in requests.get(labels_url).json()["value"]:
        key = item["Key"].strip()
        title = item["Title"].strip()
        labels[key] = title

    df["KenmerkenVerzuimgeval"] = df["KenmerkenVerzuimgeval"].str.strip()
    df["Verzuim_label"] = df["KenmerkenVerzuimgeval"].map(labels)

    return df

df_duur = load_verzuimduur()

@st.cache_data
def verzuimtijd(dframe: pd.DataFrame) -> pd.DataFrame:
    dframe = dframe[
        (dframe["Perioden"] == "2024JJ00") &
        (dframe["Verzuim_label"]).str.startswith("Klacht:") &
        (dframe["Verzuim_label"] != "Klacht: overig")
    ]
    dframe = dframe[["Gemiddeld_5","Verzuim_label"]].sort_values("Gemiddeld_5", ascending=False).head(5)
    dframe["Verzuim_stripped"] = dframe["Verzuim_label"].str.replace("Klacht: ","")
    return dframe

with st.container(border=True):
    with st.container(horizontal=True):
        df_totaal = df[(df["label_clean"] == "Alle economische activiteiten")]
        slope, intercept, r, p, se = linregress(df_totaal["Jaar"].astype(int), df_totaal["burnout_score"])

        df_totaal["trend"] = intercept + slope * df_totaal["Jaar"].astype(int)

        fig = px.line(
            df_totaal,
            x="Jaar",
            y=["trend", "burnout_score"],
            labels={
                "value": "Kans op burnout (%)",
            },
            title="Uitval door burnout, totale bevolking 2014-2024",
        )

        fig.update_traces(
            selector={"name": "trend"},
            line=dict(color="grey", dash="dot", width=1.5),
            opacity=0.85,
            hoverinfo="skip",
            hovertemplate= None
        )
        fig.update_traces(
            selector={"name": "burnout_score"},
            line=dict(color="orange", width=1.5),
            hovertemplate="<span style='font-size:16px; color:orange'><b>%{y:.2f}%</b></span><extra></extra>",
        )

        fig.update_layout(showlegend=False,hovermode="x")

        st.plotly_chart(fig)

        df_verzuimtijd = verzuimtijd(df_duur)

        fig = px.bar(
            df_verzuimtijd.sort_values("Gemiddeld_5"),
            x="Verzuim_stripped",
            y="Gemiddeld_5",
            orientation="v",
            labels={"Gemiddeld_5": "Gemiddeld Duur Verzuim (dagen)", "Verzuim_stripped": "Klacht"},
            text=df_verzuimtijd.sort_values("Gemiddeld_5")["Gemiddeld_5"].apply(lambda x: f"{x:.1f} dagen"),
            title="Klachten met langste verzuimtijd",
            color="Gemiddeld_5",
            color_continuous_scale= [(0,"orange"), (1,"red")],
        )

        fig.update_traces(
            hovertemplate="<span style='font-size:12px;text-transform:capitalize; color:white'><b>%{x}</b></span><br><span style='font-size:12px; color:orange'><b>%{y:.2f} dagen</b></span>",
        )

        fig.update(layout_coloraxis_showscale=False)

        st.plotly_chart(fig)

    with st.expander("Meer informatie"):
        st.write("De grafiek links laat zien hoe de totale kans op een burnout toe is genomen. Dit is het gemiddelde van de gehele bevolking van nederland. De kans op burnout is berekend door de kans dat iemand verzuimd te vermenigvuldigen met de kans dat de verzuimklacht burnout is. "
                 "In deze data zitten dus alleen werknemers ook daadwerkelijk zijn uitgevallen wegens burnout, geen werknemers die ondanks burnout klachten doorwerken")
        st.write("De grafiek rechts toont de gemiddelde verzuimtijd van de 5 klachten met de langste verzuimtijd. Burnout klachten laten werknemers er gemiddeld twee maanden uitliggen. Let op: dis is een gemiddelde, en wordt "
                 "(net als alle klachten in de dataset) beïnvloed door uitschieters. De mediaan voor burnout ligt op 21 dagen uitval, en zo\'n 7.3%% van werknemers [ligt er langer dan een jaar uit.](https://opendata.cbs.nl/#/CBS/nl/dataset/86168NED/table?ts=1776154285299) ")

@st.cache_data
def load_data_beroepen():
    data_url = "https://opendata.cbs.nl/ODataApi/odata/86010NED/TypedDataSet?$filter=Marges eq 'MW00000'"
    df_b = pd.DataFrame(requests.get(data_url).json()["value"])

    # Bedrijfskenmerken labels
    labels_url = "https://opendata.cbs.nl/ODataApi/odata/86010NED/Beroep"

    # Voeg leesbare labels toe uit de metadata
    labels = {}
    for item in requests.get(labels_url).json()["value"]:
        key = item["Key"].strip()
        s = item["Description"].strip().split(':')[0]
        if s and len(s) < 70:
            title = item["Description"].strip().split(':')[0]
        else:
            title = item["Title"].strip().lstrip('0123456789')
        labels[key] = title

    df_b["Beroep"] = df_b["Beroep"].str.strip()
    df_b["Beroep_label"] = df_b["Beroep"].map(labels)
    df_b["Kenmerk_type"] = df_b["Beroep_label"].apply(
        lambda x: "Beroepsniveau" if str(x).startswith("Beroepsniveau") else "Beroepsgroep"
    )

    df_b.drop_duplicates(subset=["Beroep_label","Perioden"],inplace=True)

    indices_to_drop = df_b[df_b['Beroep_label'].str.contains('overig',na=False)].index
    df_b.drop(indices_to_drop, inplace=True)

    # De waardes waren verspreid over twee kolommen
    df_b["BurnoutCombined"] = df_b["PsychischOverspannenheidBurnOut_27"].combine_first(
        df_b["PsychischOverspannenheidBurnOut_14"])

    # Jaar normaal
    df_b["Jaar"] = df_b["Perioden"].str[0:4]

    # Burnout-score: kans dat werknemer uitvalt door burnout. Delen door honderd voor percentage
    df_b["burnout_score"] = (
                                  pd.to_numeric(df_b["AandeelWerknemersDieHebbenVerzuimd_2"], errors="coerce") *
                                  pd.to_numeric(df_b["BurnoutCombined"], errors="coerce")
                          ) / 100

    # Deze gebruiken we als text label in grafieken
    df_b["burnout_label"] = df_b["burnout_score"].map("{:.2f}%".format)

    return df_b

df_b = load_data_beroepen()

@st.cache_data
def top10_burnout(df: pd.DataFrame) -> pd.DataFrame:

    #Codes die we niet mee willen nemen (aggregaten)
    aggregaat_codes = {
        "435500",
    }

    meest_recent = df["Perioden"].max()
    df_recent = df[
        (df["Perioden"] == meest_recent) &
        (df["Kenmerk_type"] == "Bedrijfstak") &
        (~df["Bedrijfskenmerken"].isin(aggregaat_codes))
    ].copy()

    result = (
        df_recent[["label_clean", "burnout_score","AandeelWerknemersDieHebbenVerzuimd_1","PsychischOverspannenheidBurnOut_26","Perioden","burnout_label"]]
        .sort_values("burnout_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    result.index += 1  # Ranking begint bij 1
    return result

df_top10 = top10_burnout(df)

@st.cache_data
def top10_burnout_beroep(df_beroep: pd.DataFrame) -> pd.DataFrame:

    meest_recent = df_beroep["Jaar"].max()
    df_recent = df_beroep[
        (df_beroep["Jaar"] == meest_recent) &
        (df_beroep["Kenmerk_type"] == "Beroepsgroep")
    ].copy()

    result = (
        df_recent[["Beroep_label", "burnout_score","AandeelWerknemersDieHebbenVerzuimd_2","PsychischOverspannenheidBurnOut_27","Jaar","burnout_label"]]
        .sort_values("burnout_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    result.index += 1  # Ranking begint bij 1
    return result

df_top10_beroep = top10_burnout_beroep(df_b)

sLong = """De kans op burnout varieert sterk per sector en beroepsgroep.
Met name zorgende beroepen (verpleegkundigen, maatschappelijk/sociaal werkers), beroepen in het onderwijs en in de overheid lijken gevoelig voor een burnout.
De gemiddeldes liggen binnen sectoren gematigder, terwijl beroepsgroepen grotere uitschieters kent. Dit is omdat beroepen binnen een hele sector een relatief grote variatie aan werkzaamheden kent. 
Niet iedereen binnen de sector onderwijs, staat ook voor de klas. Hierdoor worden deze gemiddeldes naar het midden getrokken. 
"""
st.markdown(sLong)



with st.container(horizontal=True):
    #barchart
    fig = px.bar(
        df_top10.sort_values("burnout_score"),
        x="burnout_score",
        y="label_clean",
        orientation="h",
        labels={"burnout_score": "Burnout kans (%)", "label_clean": "Sector"},
        text="burnout_label",
        title="Top 10 sectoren met hoogste kans op burnout",
        color = "burnout_score",
        color_continuous_scale=[(0, "yellow"), (1, "orange")],

    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(xaxis_ticksuffix="%")

    fig.update_traces(
        hovertemplate="<span style='font-size:12px;text-transform:capitalize; color:white'><b>%{y}</b></span><br><span style='font-size:12px; color:orange'><b>%{x:.2f}%</b></span>",
    )
    st.plotly_chart(fig,key="SectorBar")


    #barchart
    fig = px.bar(
        df_top10_beroep.sort_values("burnout_score"),
        x="burnout_score",
        y="Beroep_label",
        orientation="h",
        labels={"burnout_score": "Burnout kans (%)", "Beroep_label": "Beroep"},
        text="burnout_label",
        title="Top 10 beroepen met hoogste kans op burnout",
        color="burnout_score",
        color_continuous_scale=[(0, "orange"), (1, "red")],

    )
    fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(xaxis_ticksuffix="%")
    fig.update_traces(
        hovertemplate="<span style='font-size:12px;text-transform:capitalize; color:white'><b>%{y}</b></span><br><span style='font-size:12px; color:orange'><b>%{x:.2f}%</b></span>",
    )

    st.plotly_chart(fig,key="BeroepBar")


@st.cache_data
def get_gefilterde_dfs():
    df_s = df[df["Kenmerk_type"] != "Vestigingsgrootte"]
    df_b_f = df_b[df_b["Kenmerk_type"] != "Beroepsniveau"]
    return df_s, df_b_f

df_s_gefilterd, df_b_gefilterd = get_gefilterde_dfs()

# bs staat hier voor beroep en sector, niet bullshit

@st.cache_data
def get_combined():
    bs_combine = list(set(df_b_gefilterd["Beroep_label"]) | set(df_s_gefilterd["label_clean"]))
    bs_combine.sort()
    return bs_combine
bs_combined = get_combined()

def get_recent(df_jaar):
    """""Returns the most recent year in the dataset"""
    return df_jaar["Jaar"].max()

def get_burnoutkans(v,year=None):

    if v in set(df_b_gefilterd["Beroep_label"]):
        if year is None:
            year = get_recent(df_b)
        df_kans=df_b[
            (df_b["Jaar"]==year) &
            (df_b["Beroep_label"]==v)
        ]["burnout_score"]
    else:
        if year is None:
            year = get_recent(df)
        df_kans = df[
            (df["Jaar"] == year) &
            (df["label_clean"] == v)
            ]["burnout_score"]
    return df_kans.iloc[0]

@st.cache_data
def get_burnoutkans_full(v):
    """"Haalt de hele serie op"""
    if v in set(df_b_gefilterd["Beroep_label"]):
        df_kans=df_b[
            (df_b["Beroep_label"]==v)
        ]
    else:
        df_kans = df[
            (df["label_clean"] == v)
            ]
    df_kans.sort_values("Jaar", ascending=True, inplace=True)
    return df_kans["burnout_score"]

def format_bs(key):
    if key in set(df_b_gefilterd["Beroep_label"]):
        return "👤 " + key
    else:
        return "🏢 " + key

@st.cache_data
def filter_NaN(bs):
    """"Om degene zonder data er uit te filteren, in wrapper functie om het resultaat te bewaren met cache_data"""
    return [x for x in bs if get_burnoutkans(x) > 0]


with st.container(horizontal=True,border=True):

    bs_combined = filter_NaN(bs_combined)

    with st.container(border=False):
        v = st.selectbox("Benieuwd naar de burnout kans voor jouw beroep of sector? Selecteer deze hier:",bs_combined,format_func=format_bs)
    with st.container(border=False):
        kans = get_burnoutkans(v)
        if kans > 6:
            color = 'red'
        elif kans > 5:
            color = 'orange'
        elif kans > 3:
            color = 'yellow'
        else:
            color = 'green'

        st.metric(label=f"Kans op burnout voor :blue[{v}] is",value=f":{color}[{kans:.2f}%]",delta=str(round(kans - get_burnoutkans(v,"2023"),2))+"%",delta_color="inverse",chart_data=get_burnoutkans_full(v))


@st.cache_data
def line(df: pd.DataFrame) -> pd.DataFrame:
    top10_labels = (
        df[
            (df["Perioden"] == "2024JJ00") &
            (df["Kenmerk_type"] == "Bedrijfstak")
            ]
        .nlargest(10, "burnout_score")["Bedrijfstak_label"]
        .tolist()
    )
    top10_labels.remove('D Energievoorziening')

    # Filter alle jaren voor die 10 bedrijfstakken
    df_trend = df[
        (df["Bedrijfstak_label"].isin(top10_labels))
        ].copy()

    df_trend["label_clean"] = df_trend["Bedrijfstak_label"].str.replace(r"^[A-Z][-A-Z]* ", "", regex=True)

    return df_trend

@st.cache_data
def line_b(df: pd.DataFrame) -> pd.DataFrame:
    top10_labels = (
        df[
            (df["Perioden"] == "2024JJ00") &
            (df["Kenmerk_type"] == "Beroepsgroep")
            ]
        .nlargest(10, "burnout_score")["Beroep_label"]
        .tolist()
    )

    # Filter alle jaren voor die 10 beroepsgroepen
    df_trend = df[
        (df["Beroep_label"].isin(top10_labels))
        ].copy()

    return df_trend

st.write("Burnout klachten nemen toe voor alle sectoren en beroepen. Wel verschillen de patronen per beroepsgroep en sector. Als we kijken naar de laatste 10 jaar, zien we dat veel sectoren een vermindering"
         "van de hoeveelheid burnout klachten hadden in 2020, het jaar dat COVID-19 pandemie. Voor een sluitende conclusie waarom dit zo is, zal een diepere analyse moeten worden gedaan, maar mogelijke redenen hiervoor"
         "kunnen liggen bij de mogelijkheid om thuis te werken, of dat veel mensen hun baan uberhaupt niet konden uitvoeren. Echter, er zijn een aantal beroepen die juist een toename aan burnout kenden in 2020, zoals maatschappelijk werkers.  ")

with st.container(horizontal=False,border=True):
    st.badge(label="Tip: Dubbelklik op de verschillende sectoren/ beroepen in de legenda om die te isoleren in de grafiek.",icon=":material/info:")

    with st.container(horizontal=True,border=False):
        df_trend = line(df)
        fig = px.line(
            df_trend,
            x="Jaar",
            y="burnout_score",
            color="label_clean",
            labels={"burnout_score": "Burnout kans (%)", "Jaar": "Jaar", "label_clean": "Sector"},
            markers=True,
            title="Toename burnoutklachten in 10 bovenste sectoren",
            height=500,

        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.6,
        ))

        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Jaar: %{x}<br>Kans op burnout: %{y:.1f}%<extra></extra>"
        )

        st.plotly_chart(fig)

        df_trend = line_b(df_b)
        fig = px.line(
            df_trend,
            x="Jaar",
            y="burnout_score",
            color="Beroep_label",
            labels={"burnout_score": "Burnout kans (%)", "Jaar": "Jaar", "Beroep_label": "Beroep"},
            markers=True,
            title="Toename burnoutklachten in 10 bovenste beroepsgroepen",
            height=500,
        )
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.6,
        ))

        fig.update_traces(
            hovertemplate="<b>%{fullData.name}</b><br>Jaar: %{x}<br>Kans op burnout: %{y:.1f}%<extra></extra>"
        )

        st.plotly_chart(fig)


@st.cache_data
def load_extra_data(url, url_labels,code,new_col):
    dfr = pd.DataFrame(requests.get(url).json()["value"])

    labels = {}
    for item in requests.get(url_labels).json()["value"]:
        key = item["Key"].strip()
        title = item["Title"].strip()
        labels[key] = title

    dfr[code] = dfr[code].str.strip()
    dfr[new_col] = dfr[code].map(labels)

    if code == "Beroep":
        dfr["label_clean"] = dfr[new_col].str.lstrip("0123456789 ")
    else:
        dfr["label_clean"] = dfr[new_col].str.replace(r"^[A-Z][-A-Z]* ", "", regex=True)

    return dfr

@st.cache_data
def koppel_data(koppel_url):
    """"Om de correlatieparameters te koppelen aan hun overlappende categorie en beschrijving"""

    data = []
    parents = {}
    for item in requests.get(koppel_url).json()["value"]:
        if item["Type"] == "Topic" and item["Unit"] == "%":
            data.append({"ID": item["ID"], "ParentID": item["ParentID"],"Key": item["Key"], "Title": item["Title"],"Description": item["Description"]})
        if item["Type"] == "TopicGroup":
            parents[item["ID"]]= {"Title": item["Title"],"Description": item["Description"]}
    for item in data:
        item["Parent"] = parents[item["ParentID"]]["Title"]

    df_koppeling = pd.DataFrame(data)
    return df_koppeling, parents



urlpsycho = "https://opendata.cbs.nl/ODataApi/odata/83157NED/TypedDataSet?$filter=Marges eq 'MW00000'"
urlpsycho_labels = "https://opendata.cbs.nl/ODataApi/odata/83157NED/BedrijfstakkenBranchesSBI2008"
urlduurzaam = "https://opendata.cbs.nl/ODataApi/odata/83156NED/TypedDataSet?$filter=Marges eq 'MW00000'"
urlduurzaam_labels = "https://opendata.cbs.nl/ODataApi/odata/83156NED/BedrijfstakkenBranchesSBI2008"

urlpsycho_b = "https://opendata.cbs.nl/ODataApi/odata/84436NED/TypedDataSet"
urlpsycho_b_labels = "https://opendata.cbs.nl/ODataApi/odata/84436NED/Beroep"
urlduurzaam_b = "https://opendata.cbs.nl/ODataApi/odata/84434NED/TypedDataSet"
urlduurzaam_b_labels = "https://opendata.cbs.nl/ODataApi/odata/84434NED/Beroep"

df_psych = load_extra_data(urlpsycho,urlpsycho_labels,"BedrijfstakkenBranchesSBI2008","Bedrijfstak_label")
df_duur = load_extra_data(urlduurzaam,urlduurzaam_labels,"BedrijfstakkenBranchesSBI2008","Bedrijfstak_label")

df_psych_b = load_extra_data(urlpsycho_b,urlpsycho_b_labels,"Beroep","Beroep_label")
df_duur_b = load_extra_data(urlduurzaam_b,urlduurzaam_b_labels,"Beroep","Beroep_label")

koppel_url_psych = "https://opendata.cbs.nl/ODataApi/odata/83157NED/DataProperties"
df_koppeling_psych, parents_psych = koppel_data(koppel_url_psych)

koppel_url_duur = "https://opendata.cbs.nl/ODataApi/odata/83156NED/DataProperties"
df_koppeling_duur, parents_duur = koppel_data(koppel_url_duur)

koppel_url_psych_b = "https://opendata.cbs.nl/ODataApi/odata/84436NED/DataProperties"
df_koppeling_psych_b, parents_psych_b = koppel_data(koppel_url_psych_b)

koppel_url_duur_b = "https://opendata.cbs.nl/ODataApi/odata/84434NED/DataProperties"
df_koppeling_duur_b, parents_duur_b = koppel_data(koppel_url_duur_b)

@st.cache_data
def analyse_full(df_sector, df_beroep, df_fact_s,df_fact_b,df_koppel_s,df_koppel_b):
    """"Function made to prep/combine the data needed for the analyse function below"""

    # First, combine the sector and beroep
    df_sector = df_sector[
        (df_sector["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_sector["Kenmerk_type"] == "Bedrijfstak")
        ].copy()
    df_sector = df_sector[["burnout_score", "Perioden", "label_clean"]]

    df_beroep = df_beroep[
        (df_beroep["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_beroep["Kenmerk_type"] == "Beroepsgroep")
        ].copy()
    df_beroep["label_clean"] = df_beroep["Beroep_label"]
    df_beroep = df_beroep[["burnout_score", "Perioden", "label_clean"]]

    df_comb = pd.concat([df_sector,df_beroep])
    df_comb.dropna(inplace=True)

    # Then, combine the 'factors'
    df_fact_s = df_fact_s[
        (df_fact_s["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (~df_fact_s["BedrijfstakkenBranchesSBI2008"].str.startswith("T"))
        ].copy()


    df_fact_b = df_fact_b[
        (df_fact_b["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_fact_b["Beroep_label"].str[0].str.isdigit())
        ].copy()

    df_fact_comb = pd.concat([df_fact_s,df_fact_b])

    reasonList = []

    for p in df_koppel_s["Key"]:
        df_pred = df_fact_comb[[p, "Perioden", "label_clean"]].copy()
        df_merged = pd.merge(df_comb, df_pred, on=["Perioden", "label_clean"]).dropna(axis=0)
        try:
            corr = pearsonr(df_merged["burnout_score"].values, df_merged[p].values)[0]
            reasonList.append({"Key": p, "Correlation": corr})
        except:
            continue

    df_updated = pd.merge(df_koppel_s,pd.DataFrame(reasonList))

    return df_updated

correlation_psych = analyse_full(df,df_b,df_psych,df_psych_b,df_koppeling_psych,df_koppeling_psych_b)
correlation_duur = analyse_full(df,df_b,df_duur,df_duur_b,df_koppeling_duur,df_koppeling_duur_b)


df_correlation = pd.concat([correlation_psych,correlation_duur])

# Om makkelijk dingen op te kunnen zoeken met de key
df_correlation_indexed = df_correlation.set_index("Key")


fig = px.bar(
    df_correlation,
    x='Key',
    y='Correlation',
    labels={"Key":"Variabele","Correlation": "Correlatie"},
    color='Parent',
    height=1000,
    #range_y=[-0.5, 1],

)

fig.update_layout(
    xaxis={
        'categoryorder': 'total ascending',
        'tickmode': 'array',
        'tickvals': df_correlation['Key'],
        'ticktext': df_correlation['Title'],
        'tickangle': 40,
    },
    legend={
        'yanchor':"top",
        'y':0.99,
        'xanchor':"left",
        'x':0.01,
        'title': 'Categorie'
    }
)

fig.update_traces(
            hovertemplate="<b>%{x}</b><br>Categorie: %{fullData.name}<br>Correlatie: %{y:.2f}<extra></extra>"
        )

link_cbs = "https://opendata.cbs.nl/#/CBS/nl/navigatieScherm/thema?themaNr=83350"

sLong = """
Veel verschillende factoren kunnen bijdragen aan een burn-out.
Het CBS heeft werknemers in de eerdergenoemde sectoren en beroepsgroepen ook gevraagd naar de ervaren [arbeidsomstandigheden](%s).
We kijken hier naar de 'Psycho-sociale arbeidsbelasting' en 'Duurzame inzetbaarheid'.  
\n
Op basis van deze gegevens kan de correlatie met burn-out worden berekend. We zien dat de categorieën 'Psychische vermoeidheid door werk' en 'Emotioneel belastend werk' de variabelen bevatten
met de sterkste correlatie tot burnout. Dit is geen verassing, gezien het klachtenprofiel van burnout mentaal is.
""" % link_cbs

st.write(sLong)
with st.container(border=True):
    st.badge(label="Let op: de data is niet uitgebreid genoeg om sterke conclusies over de correlaties te trekken. Deze zijn daarom enkel indicatief. ",icon=":material/warning:",color="red")
    st.plotly_chart(fig)
    with st.expander("Meer informatie"):
        st.write("De correlaties zijn berekend aan de hand van de [Pearson Correlatiecoëfficiënt](https://www.scribbr.nl/statistiek/pearson-correlatie/). "
                 "Correlatie is een maat voor de samenhang tussen twee variabelen en ligt tussen -1 en 1. "
                 "Een waarde van +1 duidt op een sterke positieve samenhang, terwijl -1 een even sterke maar negatieve samenhang aangeeft. "
                 "Negatieve waarden betekenen dat een toename van de ene variabele gepaard gaat met een afname van de andere. "
                 "Een correlatie van 0 wijst op het ontbreken van een samenhang.")

        st.write("Wat ook opvalt, is dat lichamelijk geweld (door leidinggevenden/collega's)"
                 " een licht negatieve correlatie met burnout zou hebben. Dit impliceert dat lichamelijk geweld kan leiden tot een afname in burnout. Dit is een goed voorbeeld van dat correlatie (voor zoverre daar al sprake van is) "
                 "niet altijd causatie betekent. Waar we hier mogelijk mee te maken hebben is een [confounding variable](https://www.scribbr.com/methodology/confounding-variables/). Hoewel een sterke conclusie"
                 "niet zonder nader onderzoek getrokken kan worden, is het mogelijk dat beroepen waar lichamelijk geweld meer voorkomt ook minder burnout kent, zonder dat deze dingen per se samenhangen."
                 " Kijk bijvoorbeeld naar de horeca, een sector die relatief weinig burnout kent, maar wel een relatief hoge hoeveelheid lichamelijk geweld.  ")



@st.cache_data
def scatter(df_uitval, df_uitval_b, df_factor_s, df_factor_b, variable):
    df_factor_s = df_factor_s[
        (df_factor_s["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (~df_factor_s["BedrijfstakkenBranchesSBI2008"].str.startswith("T"))
    ].copy()

    df_factor_b = df_factor_b[
        (df_factor_b["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_factor_b["Beroep_label"].str[0].str.isdigit())
    ].copy()

    df_uitval_s = df_uitval[
        (df_uitval["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_uitval["Kenmerk_type"] == "Bedrijfstak")
    ][["burnout_score", "Perioden", "label_clean"]].copy()

    df_uitval_b = df_uitval_b[
        (df_uitval_b["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_uitval_b["Kenmerk_type"] == "Beroepsgroep")
    ].copy()
    df_uitval_b["label_clean"] = df_uitval_b["Beroep_label"]
    df_uitval_b = df_uitval_b[["burnout_score", "Perioden", "label_clean"]]

    df_comb_uitval = pd.concat([df_uitval_s, df_uitval_b])
    df_comb_factor = pd.concat([df_factor_s, df_factor_b])

    df_pred = df_comb_factor[[variable, "Perioden", "label_clean"]].dropna(subset=[variable]).drop_duplicates(
        subset=["Perioden", "label_clean"]).copy()
    df_merged = pd.merge(df_comb_uitval, df_pred, on=["Perioden", "label_clean"]).dropna()
    df_merged["Jaar"] = df_merged["Perioden"].str[0:4]


    fig = px.scatter(
        df_merged,
        x=variable,
        y="burnout_score",
        labels={"burnout_score": "Burnout kans (%)", variable: f"{format_label(variable)} (%)",
                "label_clean": "Sector/Beroep"},
        hover_name="label_clean",
        hover_data={"Jaar":True,variable:':.2f','burnout_score':':.2f'},
        color="label_clean",
        trendline="ols",
        trendline_scope="overall",
        trendline_color_override="grey"
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(
        selector=lambda t: t.name and "Trendline" in t.name,
        hoverinfo="skip",
        hovertemplate=None
    )

    for trace in fig.data:
        print(f"Trace naam: {trace.name}, Mode: {trace.mode}")


    st.plotly_chart(fig)

def format_label(key):
    return df_correlation_indexed.loc[key, "Title"]

df_correlation.sort_values("Correlation",inplace=True,ascending=False)


st.write("We kunnen ook inzoomen op individuele variabeles. Een scatterplot leent zich goed voor het visualiseren van de mogelijke correlatie. Hieronder kan je een variabele selecteren om de complete collectie"
         " van datapunten te zien, waar de burnout kans tegen de variabele wordt geplot.")


def cor_string(variabel):
    corry = df_correlation.loc[df_correlation["Key"] == variabel, "Correlation"].iloc[0]
    s = f"De correlatie van :blue[{format_label(variabel)}] en burnout is  "
    if abs(corry) > 0.5:
        s+= f" :red[{corry:.2f}]. Dit impliceert een sterke"
    elif abs(corry) > 0.29:
        s+= f" :orange[{corry:.2f}]. Dit impliceert een matig"
    else:
        s+= f" :yellow[{corry:.2f}]. Dit impliceert een zwakke tot geen correlatie tussen de variabele en de kans op burnout."
        return s
    if corry > 0:
        s+= " positieve"
    else:
        s+= " negatieve"
    s+= " correlatie tussen de variabele en de kans op burnout."
    return s


variabel = st.selectbox(label="Selecteer een variabele voor de scatter", options=df_correlation["Key"],
                            format_func=format_label)

col1, col2 = st.columns(2)
with col1:
    scatter(df, df_b, pd.concat([df_psych,df_duur]),pd.concat([df_psych_b,df_duur_b]), variabel)

with col2:
    st.space()
    st.space()
    st.write(cor_string(variabel))
    s = df_correlation_indexed.loc[variabel, "Description"].split('Antwoordcategorie')[0]
    s = s + '\n' + ':gray[Categorie: '+ df_correlation_indexed.loc[variabel, "Parent"]+']'
    st.write("Details:")
    st.info(s)



@st.cache_data
def prep_predictor(df_uitval, df_uitval_b, df_factor_s, df_factor_b,koppel):
    """"Get all data formatted for the predictor.
    Desired: X = the values of the different burnout indicators for every job/sector
    y = the burnout probability for said job/sector"""
    df_uitval_s = df_uitval[
        (df_uitval["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_uitval["Kenmerk_type"] == "Bedrijfstak")
        ][["burnout_score", "Perioden", "label_clean"]].copy()

    df_uitval_b = df_uitval_b[
        (df_uitval_b["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_uitval_b["Kenmerk_type"] == "Beroepsgroep")
        ].copy()
    df_uitval_b["label_clean"] = df_uitval_b["Beroep_label"]
    df_uitval_b = df_uitval_b[["burnout_score", "Perioden", "label_clean"]]

    df_comb_uitval = pd.concat([df_uitval_s, df_uitval_b]).dropna()

    df_factor_s = df_factor_s[
        (df_factor_s["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (~df_factor_s["BedrijfstakkenBranchesSBI2008"].str.startswith("T"))
        ].copy()

    df_factor_b = df_factor_b[
        (df_factor_b["Perioden"].isin(["2022JJ00", "2023JJ00", "2024JJ00"])) &
        (df_factor_b["Beroep_label"].str[0].str.isdigit())
        ].copy()

    df_comb_factor = pd.concat([df_factor_s, df_factor_b]).drop_duplicates(
        subset=["Perioden", "label_clean"]).copy()

    df_merged = pd.merge(df_comb_uitval, df_comb_factor, on=["Perioden", "label_clean"]).dropna(axis=1,how="all")

    features = koppel["Key"].tolist()

    df_merged = df_merged.loc[:,df_merged.columns.isin(features+["burnout_score"])].dropna()

    # To get rid of burnout score, gives us all the used features. Returned for the slider function
    features = df_merged.columns.tolist()[1:]

    y = df_merged["burnout_score"].values
    X = df_merged.loc[:,df_merged.columns.isin(features)].values

    return X, y,features

@st.cache_data
def train_predictor(X,y):
    """"Een regressie getrained op de variabeles om burnout te voorspellen.
    Ridge is gebruik om overfitting te voorkomen, gezien de hoeveelheid noise in de dataset. """
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=3)
    model.fit(X_scaled, y)
    return model, scaler


def sliders(features, X):
    Xlabeled = pd.DataFrame(X, columns=features)

    for f in features:
        if f not in st.session_state:
            st.session_state[f] = float(Xlabeled[f].mean())


    with st.container(horizontal=True):
        if st.button("🎲 Random"):
            for f in features:
                st.session_state[f] = float(np.random.uniform(Xlabeled[f].min(), Xlabeled[f].max()))
            st.rerun()
        if st.button("⚖️ Average"):
            for f in features:
                st.session_state[f] = Xlabeled[f].mean()
            st.rerun()

    parents = [p for p in df_correlation["Parent"].unique()
               if not df_correlation[(df_correlation["Parent"] == p) & (df_correlation["Key"].isin(features))].empty]

    # Tel het aantal sliders per parent
    parent_sizes = {p: len(df_correlation[(df_correlation["Parent"] == p) & (df_correlation["Key"].isin(features))])
                    for p in parents}

    # Verdeel over 4 kolommen op basis van totale hoogte
    col_totals = [0, 0, 0, 0]
    col_assignments = {0: [], 1: [], 2: [],3: []}

    for p in sorted(parent_sizes, key=parent_sizes.get, reverse=True):
        kleinste_col = col_totals.index(min(col_totals))
        col_assignments[kleinste_col].append(p)
        col_totals[kleinste_col] += parent_sizes[p]

    with st.form("sliderform", border=False):
        cols = st.columns(4)
        for col_idx, col_parents in col_assignments.items():
            with cols[col_idx]:
                for p in col_parents:
                    df_p = df_correlation[(df_correlation["Parent"] == p) & (df_correlation["Key"].isin(features))]
                    with st.expander(f":red[{p.replace("(vanaf 2022)",'')}]"):
                        for f in df_p["Key"]:
                            st.slider(
                                label= df_correlation_indexed.loc[f, "Title"],
                                min_value=float(Xlabeled[f].min()),
                                max_value=float(Xlabeled[f].max()),
                                value=float(Xlabeled[f].mean()),
                                key=f
                            )

        with st.container(horizontal=True):
            submitted = st.form_submit_button("Voorspel burnout kans")
            if submitted:
                X_input = np.array([[st.session_state[f] for f in features]])
                kans = max(0, model.predict(scaler.transform(X_input))[0])
                if kans > 6:
                    color = 'red'
                elif kans > 5:
                    color = 'orange'
                elif kans > 3:
                    color = 'yellow'
                else:
                    color = 'green'
                st.markdown(f"#### Voorspelde kans op burnout voor dit beroep: :{color}[{kans:.2f}%]")




X,y, features = prep_predictor(df, df_b, pd.concat([df_psych,df_duur]),pd.concat([df_psych_b,df_duur_b]),df_koppeling_psych)
model, scaler = train_predictor(X,y)

st.subheader("Burnout voorspellen")
st.write("Aan de hand van de correlaties tussen deze variabelen en burnout kunnen we een model trainen om de kans op burnout te voorspellen voor onbekende beroepsgroepen / sectoren."
         " Hieronder kan je verschillende variabelen aanpassen, en zo kijken wat het effect ervan is op de kans op burnout. De waardes staan standaard op hun gemiddeldes van de dataset, en "
         " de limieten zijn ook dezelfde als in de dataset. "
         )
sliders(features,X)

with st.expander("Meer informatie"):
    st.write("Voor het voorspellen gebruiken we een [Ridge Regression](https://www.ibm.com/think/topics/ridge-regression) model. Dit is een multidimensionaal regressiemodel dat gebruikt wordt in scenarios"
    " met veel variabelen. De kracht van het model is het voorkomen van 'overfitting' (het te sterk afstemmen van het model op de trainingsdata).")











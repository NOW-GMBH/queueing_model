"""
This script analyses and visualizes the traffic counts of several BAST-Zählstellen
"""

import calendar
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.rcParams["figure.figsize"] = (15, 8)
plt.style.use("ggplot")

BASE_URL_hourly = "https://www.bast.de/videos/{year}_{typ}_S.zip"


def load_bast_jawe(year: int = 2023, cache_dir: Path = Path(".cache")) -> pd.DataFrame:
    """
    Loads the yearly statistics of traffic count data from BASt
    :param year: year of Dataset
    :param cache_dir:
    :return:
    """

    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"Jawe{year}.csv"

    if cache_file.exists():
        return pd.read_csv(cache_file, sep=";", encoding="latin-1", low_memory=False)

    url = (
        f"https://www.bast.de/DE/Themen/Digitales/HF_1/Massnahmen/"
        f"verkehrszaehlung/Daten/{year}_1/Jawe{year}.csv"
        "?view=renderTcDataExportCSV"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = "latin-1"

    cache_file.write_text(response.text, encoding="latin-1")
    return pd.read_csv(cache_file, sep=";", encoding="latin-1", low_memory=False)


def load_bast_hourly(
    year: int, strassenklasse: str = "BAB", cache_dir: Path = Path(".cache")
) -> pd.DataFrame:
    """
    Loads the hourly statistics of traffic count data from BASt
    year: year of Dataset
    strassenklasse: 'BAB' (Autobahnen) or 'B' (Bundesstraßen)
    """
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{year}_{strassenklasse}_S.zip"

    if cache_file.exists():
        return pd.read_csv(
            cache_file,
            compression="zip",
            sep=";",
            encoding="latin-1",
            thousands=".",
            decimal=",",
            low_memory=False,
        )

    typ = "A" if strassenklasse == "BAB" else "B"
    url = BASE_URL_hourly.format(year=year, typ=typ)
    print(f"Lade: {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    response.encoding = "latin-1"

    content = response.content
    cache_file.write_bytes(content)
    return pd.read_csv(
        cache_file,
        compression="zip",
        sep=";",
        encoding="latin-1",
        thousands=".",
        decimal=",",
        low_memory=False,
    )


def parse_date(df):
    df.copy()
    df["Date"] = pd.to_datetime(
        df["Datum"].astype("str") + " " + (df["Stunde"] - 1).astype("str").str.zfill(2),
        format="%y%m%d %H",
    )
    df["Month"] = df["Date"].dt.month
    df["Weekday"] = df["Date"].dt.weekday
    df["Hour"] = df["Date"].dt.hour
    return df


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def q90(x):
    return x.quantile(0.90)


def specs(x, **kwargs):
    # plt.axhline(x.mean(), c='grey', ls='-', lw=2.5, label='Mean')
    plt.axhline(x.median(), c="grey", ls="--", lw=2.5, label="Median")


# %%

# %%

bast_hourly = load_bast_hourly(year=2020)

# %%
Zählstelle = [3937, 5090, 9529, 6963, 9028]
Zahlstellen_Name = {
    3937: "Magdeburg Kannenstieg A2 - Typ C",
    5090: "Düsseldorf-Urdenbach A59 - Typ A",
    9529: "AS Wasserlosen A7 - Typ E",
    6963: "Hochstadt A66 - Typ F",
    9028: "Felden (O) A8 - Typ G",
}

# %%
selected_bast_hourly = parse_date(bast_hourly.loc[bast_hourly["Zst"].isin(Zählstelle)])
selected_bast_hourly["Zst_Name"] = selected_bast_hourly["Zst"].map(Zahlstellen_Name)

# %%
f = ["median", "std", q1, q3, q90]
quantiles = (
    selected_bast_hourly.groupby(["Zst", "Weekday"])["PLZ_R2"].agg(f).reset_index()
)

# %%
# Tagesganglinien für ausgewählte Zählstationen mit Tagesgangtypen
tagesganglinie = sns.catplot(
    x="Hour",
    y="PLZ_R2",
    data=selected_bast_hourly[
        (selected_bast_hourly["Zst"].isin(Zählstelle))
        & (selected_bast_hourly["Weekday"].isin([1, 2, 3]))
    ],
    col="Zst_Name",
    col_wrap=2,
    hue="Weekday",
    kind="point",
    estimator=lambda x: np.percentile(x, 90),
    errorbar=None,
    legend=False,
    height=6,
    aspect=1.3,
)
tagesganglinie.set(ylim=(0, 4000))
tagesganglinie.map(specs, "PLZ_R2")
tagesganglinie.set_axis_labels("Stunde", "Anzahl Fahrzeuge (PKW Gruppe) R2")
# sns.move_legend(tagesganglinie, "center left", bbox_to_anchor=(1.05, 1.05))
hue_labels = ["Dienstag", "Mittwoch", "Donnerstag", "Median", "Mean"]
tagesganglinie.add_legend(
    legend_data={
        key: value
        for key, value in zip(hue_labels, tagesganglinie._legend_data.values())
    }
)
for ax in tagesganglinie.axes:
    plt.setp(ax.get_xticklabels(), visible=True)
# plt.tight_layout()
plt.show()
# %%
# Für Zählstelle mit stark ausgeprägtem Jahresganglinientyp G - Urlaub
zaehlstelle_g = 9028
month_order = [calendar.month_name[m] for m in range(1, 13)]

plot_data = selected_bast_hourly[
    selected_bast_hourly["Zst"].isin([zaehlstelle_g])
].copy()
plot_data["Month"] = pd.Categorical(
    plot_data["Month"].map(lambda m: calendar.month_name[m]),
    categories=month_order,
    ordered=True,
)
tagesganglinie = sns.catplot(
    x="Hour",
    y="PLZ_R1",
    data=plot_data,
    hue="Month",
    hue_order=month_order,
    palette=sns.color_palette("tab20", n_colors=12),
    kind="point",
    estimator=lambda x: np.percentile(x, 90),
    errorbar=None,
    legend=True,
    height=5,
    aspect=1.3,
)
tagesganglinie.set_axis_labels("Stunde", "Anzahl Fahrzeuge (PKW Gruppe) R1")
# sns.move_legend(tagesganglinie, "center left", bbox_to_anchor=(1.05, 1.05))

tagesganglinie.figure.subplots_adjust(right=0.75)

legend = tagesganglinie.figure.legends[0]
legend.set_title("Monat")
legend.set_bbox_to_anchor((0.78, 0.5), transform=tagesganglinie.figure.transFigure)

tagesganglinie.figure.legends.clear()

# Neu auf der Axes platzieren – außerhalb
ax = tagesganglinie.axes[0][0]
handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles=handles,
    labels=labels,
    title="Monat",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0,
)

plt.show()

# %%
for zst in Zählstelle:

    tagesganglinie = sns.catplot(
        x="Hour",
        y="PLZ_R2",
        data=selected_bast_hourly[
            (selected_bast_hourly["Zst"].isin([zst]))
            & (selected_bast_hourly["Weekday"].isin([1, 2, 3]))
        ],
        hue="Weekday",
        kind="point",
        estimator=lambda x: np.percentile(x, 90),
        ci=None,
        legend=False,
        height=6,
        aspect=1.3,
    )
    tagesganglinie.set(ylim=(0, 4000), title=Zahlstellen_Name[zst])
    tagesganglinie.map(specs, "PLZ_R2")
    tagesganglinie.set_axis_labels("Stunde", "Anzahl Fahrzeuge (PKW Gruppe) R2")
    # sns.move_legend(tagesganglinie, "center left", bbox_to_anchor=(1.05, 1.05))
    hue_labels = ["Dienstag", "Mittwoch", "Donnerstag", "Median", "Mean"]
    tagesganglinie.add_legend(
        legend_data={
            key: value
            for key, value in zip(hue_labels, tagesganglinie._legend_data.values())
        }
    )
    # plt.tight_layout()
    plt.show()

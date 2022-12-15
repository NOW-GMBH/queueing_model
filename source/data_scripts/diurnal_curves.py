#%%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['figure.figsize'] = (15,8)
plt.style.use('ggplot')

def parse_date(df):
    df.copy()
    df['Date'] = pd.to_datetime(df['Datum'].astype('str') + ' ' + (df['Stunde']-1).astype('str').str.zfill(2), format="%y%m%d %H")
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['Hour'] = df['Date'].dt.hour
    return df

def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)

def q90(x):
    return x.quantile(0.90)

def specs(x, **kwargs):
    #plt.axhline(x.mean(), c='grey', ls='-', lw=2.5, label='Mean')
    plt.axhline(x.median(), c='grey', ls='--', lw=2.5, label='Median')

#%%

#%%
bast_hourly = pd.read_csv("C:/SynologyDrive/D.020.001_BASt_Zaehlstellen/10_Jawe2020/2020_A_S.txt", sep=';')
#%%
Zählstelle = [3937, 5090, 9529, 6963, 9028]
Zahlstellen_Name = {3937: 'Magdeburg Kannenstieg A2 - Typ C', 5090: 'Düsseldorf-Urdenbach A59 - Typ A',
                    9529: 'AS Wasserlosen A7 - Typ E', 6963: 'Hochstadt A66 - Typ F', 9028: 'Felden (O) A8'}

#%%
selected_bast_hourly = parse_date(bast_hourly.loc[bast_hourly['Zst'].isin(Zählstelle)])
selected_bast_hourly['Zst_Name'] = selected_bast_hourly['Zst'].map(Zahlstellen_Name)

#%%
f = ['median', 'std', q1, q3, q90]
quantiles = selected_bast_hourly.groupby(['Zst', 'Weekday'])['PLZ_R2'].agg(f).reset_index()

#%%
# Tagesganglinien für ausgewählte Zählstationen mit Tagesgangtypen
tagesganglinie = sns.catplot(x='Hour', y='PLZ_R2',
                             data=selected_bast_hourly[(selected_bast_hourly['Zst'].isin([3937, 5090, 9529, 6963]))
                                                         & (selected_bast_hourly['Weekday'].isin([1,2,3]))],
                             col='Zst_Name', col_wrap=2, hue='Weekday', kind='point',
                             estimator= lambda x: np.percentile(x, 90), ci=None, legend=False, height=6, aspect=1.3)
tagesganglinie.set(ylim=(0, 4000))
tagesganglinie.map(specs, 'PLZ_R2')
tagesganglinie.set_axis_labels('Stunde', 'Anzahl Fahrzeuge (PKW Gruppe) R2')
#sns.move_legend(tagesganglinie, "center left", bbox_to_anchor=(1.05, 1.05))
hue_labels = ['Dienstag', 'Mittwoch', 'Donnerstag', 'Median', 'Mean']
tagesganglinie.add_legend(legend_data={
    key: value for key, value in zip(hue_labels, tagesganglinie._legend_data.values())
})
for ax in tagesganglinie.axes:
    plt.setp(ax.get_xticklabels(), visible=True)
#plt.tight_layout()
plt.show()
#%%
# Für Zählstelle mit stark ausgeprägtem Jahresganglinientyp G - Urlaub
tagesganglinie = sns.catplot(x='Hour', y='PLZ_R1',
                             data=selected_bast_hourly[(selected_bast_hourly['Zst'].isin([9028]))],
                             hue='Month', kind='point',
                             estimator= lambda x: np.percentile(x, 90), ci=None, legend=False, height=5, aspect=1.3)
tagesganglinie.set_axis_labels('Stunde', 'Anzahl Fahrzeuge (PKW Gruppe) R1')
#sns.move_legend(tagesganglinie, "center left", bbox_to_anchor=(1.05, 1.05))
hue_labels = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']
tagesganglinie.add_legend(legend_data={
    key: value for key, value in zip(hue_labels, tagesganglinie._legend_data.values())
})
#plt.tight_layout()
plt.show()
#%%
for zst in Zählstelle:

    tagesganglinie = sns.catplot(x='Hour', y='PLZ_R2',
                                 data=selected_bast_hourly[(selected_bast_hourly['Zst'].isin([zst]))
                                                             & (selected_bast_hourly['Weekday'].isin([1,2,3]))]
                                 ,hue='Weekday', kind='point',
                                 estimator= lambda x: np.percentile(x, 90), ci=None, legend=False, height=6, aspect=1.3)
    tagesganglinie.set(ylim=(0, 4000), title=Zahlstellen_Name[zst])
    tagesganglinie.map(specs, 'PLZ_R2')
    tagesganglinie.set_axis_labels('Stunde', 'Anzahl Fahrzeuge (PKW Gruppe) R2')
    #sns.move_legend(tagesganglinie, "center left", bbox_to_anchor=(1.05, 1.05))
    hue_labels = ['Dienstag', 'Mittwoch', 'Donnerstag', 'Median', 'Mean']
    tagesganglinie.add_legend(legend_data={
        key: value for key, value in zip(hue_labels, tagesganglinie._legend_data.values())
    })
    #plt.tight_layout()
    plt.show()

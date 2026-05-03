## [LOAD PACKAGES]
import pandas as pd
import requests
import numpy as np
import statsmodels.api as stats
from scipy.stats import norm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Probit

## [API KEY]
API_key = "enter valid api key here"


## [FETCH DATA]

# Variables:
    # Household Weight: WGTP
    # Person Weight: PWGTP
    # Age: AGEP
    # Gender: SEX
    # Marital Status: MAR
    # Education Level: SCHL
    # Hours Worked Weekly: WKHP
    # Weeks Worked: WKWN
    # Wages: WAGP
    # Household Income: HINCP
    # Industry: INDP
    # Employment Status: ESR
    # Food Stamp Reciept: FS
    # Household ID: SERIALNO
    # Elder in Household: R65
    # Number of Own Children: NOC
    # Number of Other Children: NRC
    # Gave birth in past year: FER
    # Region: PUMA
    # Adjustment for income: ADJINC
# Data:
    # ACS 1-Year Estimates - Puerto Rico Public Use Microdata Sample 2024
    # Via US Census API Call

url = "https://api.census.gov/data/2024/acs/acs1/pumspr"

vars = {
    "get": "PWGTP,WGTP,AGEP,SEX,MAR,HISP,RAC1P,CIT,SCHL,WKHP,WKWN,WAGP,HINCP,INDP,ESR,FS,SERIALNO,R65,NOC,NRC,FER,PUMA,ADJINC,HUPAOC,HUPARC",
    "for": "state:72",
    "key" : API_key
}

variables = requests.get(url, params=vars)
if variables.status_code != 200:
    raise ValueError(f"There was an issue processing the US Census API request with code {variables.status_code}.")

arrays = variables.json()

df = pd.DataFrame(arrays[1:], columns=arrays[0])

## [CREATE MAIN VARIABLES]

#get household size (people with same SERIALNO are in same household)
df['HHSIZE'] = df.groupby('SERIALNO')['SERIALNO'].transform('count')

# Make ESR into binary variable (1 = Employed, 0 = Unemployed)
df['ESR'] = df['ESR'].astype(int)
df['ESR'] = df['ESR'].map({0:np.nan, 1:1, 2:1, 3:0, 4:1, 5:1, 6:np.nan})

#age to int
df['AGEP'] = df['AGEP'].astype(int)

#age squared
df['SQ_AGEP'] = np.square(df['AGEP'])

#Make SEX into binary variable (1 = Male, 0 = Female)
df['SEX'] = df['SEX'].astype(int)
df['SEX'] = df['SEX'].replace(2, 0)

#make SCHL into years of school
df['SCHL'] = df['SCHL'].astype(int)

#education levels
df['EDU1'] = (df['SCHL'] < 16).astype(int)
df['EDU2'] = ((df['SCHL'] == 16) | (df['SCHL'] == 17)).astype(int)
df['EDU3'] = ((df['SCHL'] == 18) | (df['SCHL'] == 19) | (df['SCHL'] == 20)).astype(int)
df['EDU4'] = (df['SCHL'] == 21).astype(int)
df['EDU5'] = ((df['SCHL'] == 22) | (df['SCHL'] == 23) | (df['SCHL'] == 24)).astype(int)

#interact edu2 with age (below 21 and above 21)
df['BLW21'] = (df['AGEP'] < 21).astype(int)
df['EDU_AGE'] = df['BLW21'] * df['EDU2']

#marriage dummy variable
df['MAR'] = df['MAR'].astype(int)
df['MARRIED'] = (df['MAR']== 1).astype(int)

#individual wage/salary income
df['WAGP'] = df['WAGP'].astype(float)
df['ADJINC'] = df['ADJINC'].astype(float)

#adjusted wagp
df['ADJ_WAGP'] = df['WAGP'] * df['ADJINC']

#remove -60,000 income observations
df['HINCP'] = df['HINCP'].astype(float)
df['HINCP'] = df['HINCP'].replace(-60000, np.nan)

#adjusted houeshold income
df['ADJ_HINCP'] = df['HINCP'] * df['ADJINC']

#industry variables
indp_json = pd.read_json("INDP24.json")

indp = indp_json['values']['item']
df['INDP_cat'] = df['INDP'].astype(str).map(indp)
df['INDP_prefix'] = df['INDP_cat'].str.split('-').str[0]

#map prefix to value
pre = sorted(df['INDP_prefix'].dropna().unique())
pmap = {p: i+1 for i, p in enumerate(pre)}
df['INDP_code'] = df['INDP_prefix'].map(pmap)

df_mapping = df[['INDP_code', 'INDP_prefix']].dropna().drop_duplicates()

#categorized industries
df['IND1'] = ((df['INDP_code'] == 2) | (df['INDP_code'] == 3) | (df['INDP_code'] == 18) | (df['INDP_code'] == 6)).astype(int)
df['IND2'] = (df['INDP_code'] == 10).astype(int)
df['IND3'] = ((df['INDP_code'] == 20) | (df['INDP_code'] == 14) | (df['INDP_code'] == 17)).astype(int)
df['IND4'] = ((df['INDP_code'] == 8) | (df['INDP_code'] == 7)).astype(int)
df['IND5'] = ((df['INDP_code'] == 13) | (df['INDP_code'] == 1) | (df['INDP_code'] == 16)).astype(int)
df['IND6'] = ((df['INDP_code'] == 4) | (df['INDP_code'] == 9) | (df['INDP_code'] == 15) | (df['INDP_code'] == 5) | (df['INDP_code'] == 11)).astype(int)

#FS to binary variable (Recieved = 1 , Didn't = 0)
df['FS'] = df['FS'].astype(int)
df['FS'] = df['FS'].replace(2, 0)

#number of own children
df['NOC'] = df['NOC'].astype(int)
df['NOC'] = df['NOC'].replace(-1, np.nan)

#number of own children under 6
df['HUPAOC'] = df['HUPAOC'].astype(int)
df['OC_under6'] = ((df['HUPAOC'] == 1) | (df['HUPAOC'] == 3)).astype(int)

#whether or not gave birth in past 12 months
df['FER'] = df['FER'].astype(int)
df['FER'] = df['FER'].replace(2, 0)

#interact FER, NOC, OC_under6 with Married
df['FER_MAR'] = df['FER'] * df['MARRIED']
df['NOC_MAR'] = df['NOC'] * df['MARRIED']
df['OCunder6_MAR'] = df['OC_under6'] * df['MARRIED']


## [UNEMPLOYMENT RATE BY PUMA REGION]

#00100: Aguadilla, Aguada, Moca, & Rincon
oo100_24 = (1258+963+848+804) / (16683+13780+12125+5743) * 100
oo100_23 = (1182+920+873+666) / (16742+13837+12238+5660) * 100
oo100_22 = (1417+955+1032+672) / (16928+13838+12306+5654) * 100

#00200: Mayaguez, Anasco & Hormigueros
oo200_24 = (1762+324+665) / (24796+6132+10375) * 100
oo200_23 = (1653+267+609) / (24883+6085+10394) * 100
oo200_22 = (1739+292+646) / (25087+6176+10435) * 100

#00300: Cabo Rojo, San German & Lajas
oo300_24 = (929+875+565) / (15367+10790+5900) * 100
oo300_23 = (1006+809+484) / (15576+10741+5806) * 100
oo300_22 = (1041+773+534) / (15632+10750+5869) * 100

#00400: Yauco, Sabana Grande, Penuelas, Guayanilla & Guanica
oo400_24 = (1314+531+707+595+651) / (10861+6538+6430+5394+4443) * 100
oo400_23 = (1140+484+655+618+471) / (10660+6565+6386+5488+4252) * 100
oo400_22 = (1228+514+635+584+376) / (10834+6604+6360+5413+4225) * 100

#00500: Ponce
oo500_24 = (3356) / (49893) * 100
oo500_23 = (3414) / (50157) * 100
oo500_22 = (3503) / (50158) * 100

#00600: Juana Diaz, Coamo, Villalba, & Santa Isabel
oo600_24 = (1136+821+773+641) / (17283+11573+7933+8942) * 100
oo600_23 = (1271+873+809+691) / (17654+11616+7978+8887) * 100
oo600_22 = (1280+870+942+723) / (17550+11645+8120+8648) * 100

#00700: Guayama, Salinas, Patillas, Arroyo, & Maunabo
oo700_24 = (919+692+385+473+317) / (12133+7918+4524+5152+3149) * 100
oo700_23 = (862+689+442+498+398) / (11792+7702+4492+5064+3203) * 100
oo700_22 = (837+703+439+448+349) / (11996+7776+4568+5114+3165) * 100

#00800: Humacao, Fajardo, Naguabo, Yabucoa, Ceiba, Culebra & Vieques
oo800_24 = (1272+870+636+763+292+17+151) / (19610+11903+9345+9591+4040+1100+2781) * 100
oo800_23 = (1366+895+593+777+284+22+129) / (19932+11817+9235+9506+3998+1071+2787) * 100
oo800_22 = (1254+740+536+771+253+22+138) / (19581+11963+9189+9504+3972+997+2780) * 100

#00900: Rio Grande, Canovanas, Loiza, & Luquillo
oo900_24 = (993+716+414+412) / (18964+18140+9179+6863) * 100
oo900_23 = (951+802+462+458) / (18819+18067+9139+6845) * 100
oo900_22 = (962+911+488+409) / (18869+18228+9192+6816) * 100

#01000: Carolina
o1000_24 = (2249) / (67761) * 100
o1000_23 = (2714) / (67588) * 100
o1000_22 = (3044) / (68124) * 100

#01101 & 01102: San Juan
o1100_24 = (5186) / (148636) * 100
o1100_23 = (6317) / (148496) * 100
o1100_22 = (6655) / (149038) * 100

#01200: Guaynabo & Catano
o1200_24 = (1004+295) / (42424+8541) * 100
o1200_23 = (1117+353) / (42143+8521) * 100
o1200_22 = (1239+409) / (42302+8597) * 100

#01300: Bayamon
o1300_24 = (2639) / (71834) * 100
o1300_23 = (3225) / (71755) * 100
o1300_22 = (3415) / (72177) * 100

#01400: Toa Baja & Dorado
o1400_24 = (1084+785) / (32097+15347) * 100
o1400_23 = (1304+704) / (32045+15119) * 100
o1400_22 = (1454+628) / (32315+15053) * 100

#01500: Toa Alta, Corozal & Naranjito
o1500_24 = (1099+831+463) / (31015+9991+7836) * 100
o1500_23 = (1262+618+509) / (30887+9707+7812) * 100
o1500_22 = (1338+705+551) / (31060+9838+7881) * 100

#01600: Vega Baga, Vega Alta & Morovis
o1600_24 = (929+575+646) / (14432+10858+9104) * 100
o1600_23 = (1020+564+646) / (14544+10725+8995) * 100
o1600_22 = (1052+736+619) / (14668+10944+8987) * 100

#01700: Arecibo, Manati, Barceloneta, & Florida
o1700_24 = (1599+665+451+266) / (27495+12494+6542+3634) * 100
o1700_23 = (1893+702+513+287) / (27595+12414+6543+3625) * 100
o1700_22 = (1737+740+521+288) / (26819+12524+6575+3638) * 100

#01800: Isabela, Hatillo, Camuy, & Quebradillas
o1800_24 = (792+816+619+439) / (13576+15357+11035+6886) * 100
o1800_23 = (877+880+735+480) / (13794+15334+11126+6896) * 100
o1800_22 = (929+1768+736+558) / (13802+15903+10859+6850) * 100

#01900: San Sebastian, Utuado, Lares, Adjuntas, Ciales, Jayuya, Las Marias, Maricao, & Orocovis
o1900_24 = (1201+592+760+395+331+302+331+232+402) / (11916+7612+8040+4717+4141+4806+3189+2222+5378) * 100
o1900_23 = (977+669+713+380+324+406+218+209+426) / (11683+7769+8064+4721+4064+4922+2981+2235+5375) * 100
o1900_22 = (1184+608+920+415+310+441+380+209+477) / (11844+7663+8262+4716+4052+4753+2880+2189+5433) * 100

#02000: Cayey, Cidra, Barranquitas, Aguas Buenas, Aibonito, & Comerio
o2000_24 = (855+697+551+400+487+346) / (18113+17219+8202+7225+7107+5479) * 100
o2000_23 = (946+895+589+482+495+400) / (18032+17263+8156+7284+7088+5489) * 100
o2000_22 = (863+815+604+426+488+384) / (17226+17226+8170+7271+7062+5491) * 100

#02100: Caguas
o2100_24 = (2469) / (53642) * 100
o2100_23 = (2603) / (53248) * 100
o2100_22 = (2498) / (53135) * 100

#02200: Trujillo Alto & Gurabo
o2200_24 = (634+770) / (29993+21018) * 100
o2200_23 = (782+889) / (29873+20936) * 100
o2200_22 = (942+854) / (30109+20992) * 100

#02300:San Lorenzo, Juncos & Las Piedras
o2300_24 = (721+775+868) / (12919+13891+12903) * 100
o2300_23 = (813+846+880) / (12875+12879+13806) * 100
o2300_22 = (782+857+756) / (12877+13863+12791) * 100

mapregions = {
    '00100': oo100_24,
    '00200': oo200_24,
    '00300': oo300_24,
    '00400': oo400_24,
    '00500': oo500_24,
    '00600': oo600_24,
    '00700': oo700_24,
    '00800': oo800_24,
    '00900': oo900_24,
    '01000': o1000_24,
    '01101': o1100_24,
    '01102': o1100_24,
    '01200': o1200_24,
    '01300': o1300_24,
    '01400': o1400_24,
    '01500': o1500_24,
    '01600': o1600_24,
    '01700': o1700_24,
    '01800': o1800_24,
    '01900': o1900_24,
    '02000': o2000_24,
    '02100': o2100_24,
    '02200': o2200_24,
    '02300': o2300_24
}

df['PUMA'] = df['PUMA'].astype(str)

df['unemp_reg'] = (df['PUMA']).map(mapregions)


## [NUMBER OF FS ELIGIBLE STORES / POPULATION COUNT]

#00100: Aguadilla, Aguada, Moca, & Rincon
oo100_FS = (71 + 51 + 56 + 13) / (53557+37523+37543+15625) * 100

#00200: Mayaguez, Anasco & Hormigueros
oo200_FS = (88+32+12) / (69044+24689+15290) * 100

#00300: Cabo Rojo, San German & Lajas
oo300_FS = (41+24+21) / (46820+30980+22888) * 100

#00400: Yauco, Sabana Grande, Penuelas, Guayanilla & Guanica
oo400_FS = (38+17+32+47+16) / (32057+22183+19453+16628+12117) * 100

#00500: Ponce
oo500_FS = (137) / (129659) * 100

#00600: Juana Diaz, Coamo, Villalba, & Santa Isabel
oo600_FS = (48+39+30+24) / (46045+33514+21197+19651) * 100

#00700: Guayama, Salinas, Patillas, Arroyo, & Maunabo
oo700_FS = (31+30+20+17+15) / (34386+24504+15216+14944+10248) * 100

#00800: Humacao, Fajardo, Naguabo, Yabucoa, Ceiba, Culebra & Vieques
oo800_FS = (48+33+20+27+10+4+14) / (49616+31043+22820+28580+10783+1759+7966) * 100

#00900: Rio Grande, Canovanas, Loiza, & Luquillo
oo900_FS = (32+37+24+15) / (45297+41512+22047+17387) * 100

#01000: Carolina
o1000_FS = (84) / (150679) * 100

#01101 & 01102: San Juan
o1100_FS = (256) / (332454) * 100

#01200: Guaynabo & Catano
o1200_FS = (47+27) / (89405+21971) * 100

#01300: Bayamon
o1300_FS = (143) / (181152) * 100

#01400: Toa Baja & Dorado
o1400_FS = (41+26) / (71332+35867) * 100

#01500: Toa Alta, Corozal & Naranjito
o1500_FS = (39+34+31) / (66100+34476+29516) * 100

#01600: Vega Baga, Vega Alta & Morovis
o1600_FS = (52+28+37) / (53709+34658+28230) * 100

#01700: Arecibo, Manati, Barceloneta, & Florida
o1700_FS = (70+35+24+16) / (85971+38637+22413+11425) * 100

#01800: Isabela, Hatillo, Camuy, & Quebradillas
o1800_FS = (41+54+31+25) / (43033+38034+32765+23301) * 100

#01900: San Sebastian, Utuado, Lares, Adjuntas, Ciales, Jayuya, Las Marias, Maricao, & Orocovis
o1900_FS = (71+32+52+42+20+29+11+6+36) / (38999+27155+17981+16698+14440+8674+4486+21260) * 100

#02000: Cayey, Cidra, Barranquitas, Aguas Buenas, Aibonito, & Comerio
o2000_FS = (36+31+70+21+21+32) / (40389+39478+23125+24751+18613) * 100

#02100: Caguas
o2100_FS = (99) / (124628) * 100

#02200: Trujillo Alto & Gurabo
o2200_FS = (29+20) / (66816+40039) * 100

#02300:San Lorenzo, Juncos & Las Piedras
o2300_FS = (30+36+36) / (37363+36796+34890) * 100


mapregions = {
    '00100': oo100_FS,
    '00200': oo200_FS,
    '00300': oo300_FS,
    '00400': oo400_FS,
    '00500': oo500_FS,
    '00600': oo600_FS,
    '00700': oo700_FS,
    '00800': oo800_FS,
    '00900': oo900_FS,
    '01000': o1000_FS,
    '01101': o1100_FS,
    '01102': o1100_FS,
    '01200': o1200_FS,
    '01300': o1300_FS,
    '01400': o1400_FS,
    '01500': o1500_FS,
    '01600': o1600_FS,
    '01700': o1700_FS,
    '01800': o1800_FS,
    '01900': o1900_FS,
    '02000': o2000_FS,
    '02100': o2100_FS,
    '02200': o2200_FS,
    '02300': o2300_FS
}

df['fs_availibility'] = df['PUMA'].map(mapregions)


## [LAG NAP RECEIPT BY REGION]

oo100_23 = 547
oo200_23 = 568
oo300_23 = 362
oo400_23 = 568
oo500_23 = 755
oo600_23 = 511
oo700_23 = 441
oo800_23 = 681
oo900_23 = 553
o1000_23 = 386
o1100_23 = (484+512)
o1200_23 = 318
o1300_23 = 792
o1400_23 = 381
o1500_23 = 418
o1600_23 = 551
o1700_23 = 647
o1800_23 = 507
o1900_23 = 1026
o2000_23 = 836
o2100_23 = 392
o2200_23 = 239
o2300_23 = 424

mapregions = {
    '00100': oo100_23,
    '00200': oo200_23,
    '00300': oo300_23,
    '00400': oo400_23,
    '00500': oo500_23,
    '00600': oo600_23,
    '00700': oo700_23,
    '00800': oo800_23,
    '00900': oo900_23,
    '01000': o1000_23,
    '01101': o1100_23,
    '01102': o1100_23,
    '01200': o1200_23,
    '01300': o1300_23,
    '01400': o1400_23,
    '01500': o1500_23,
    '01600': o1600_23,
    '01700': o1700_23,
    '01800': o1800_23,
    '01900': o1900_23,
    '02000': o2000_23,
    '02100': o2100_23,
    '02200': o2200_23,
    '02300': o2300_23
}

df['fs_lagreceipt'] = df['PUMA'].map(mapregions)


## [GET AVG_HHINCOME]

df['PWGTP'] = df['PWGTP'].astype(int)
df_inc = df.dropna(subset = 'ADJ_HINCP')
vals = df_inc.groupby(['PUMA', 'SEX', 'INDP_code']).apply(lambda group: np.average(group['ADJ_WAGP'], weights=group['PWGTP'])).reset_index(name='mean_WAGP')

vals['PUMA'] = vals['PUMA'].astype('Int64')
df['PUMA'] = df['PUMA'].astype('Int64')

mapping = dict(zip(zip(vals['PUMA'], vals['SEX'], vals['INDP_code']), vals['mean_WAGP']))

df['avg_WAGP'] = pd.Series(list(zip(df['PUMA'], df['SEX'], df['INDP_code']))).map(mapping)

df['avg_HINCP'] = df.groupby('SERIALNO')['avg_WAGP'].transform('sum')


## [GET SIMBENEFIT]

NAPmax_benefit_schedule = {1:183, 2:337, 3:482, 4:612, 5:727, 6:873, 7:965,
                        8:1103, 9:1241, 10:1379, 11:1517, 12:1655,
                        13:1792, 14:1930, 15:2068, 16:2206, 17:2344, 18:2482}
NAPmax_eligincome_schedule = {1:619, 2:1179, 3:1706, 4:2206, 5:2666, 6:3199, 7:3599,
                        8:4112, 9:4632, 10:5152, 11:5666, 12:6186,
                        13:6699, 14:7219, 15:7732, 16:8252, 17:8766, 18:9286}

SNAPmax_benefit_schedule = {1:291, 2:535, 3:766, 4:973, 5:1155, 6:1386,
                            7:1532, 8:1751, 9:1970, 10:2189, 11:2408, 12:2627,
                            13:2846, 14:3065, 15:3284, 16:3503, 17:3722, 18:3941}

SNAPmax_eligincome_schedule = {1:1215, 2:1644, 3:2072, 4:2500, 5:2929, 6:3357,
                               7:3785, 8:4214, 9:4643, 10:5072, 11:5501, 12:5930,
                               13:6359, 14:6788, 15:7217, 16:7646, 17:8075, 18:8504}

#CPI values by year
CPI = {2023:304.702, 2024:313.689}

'''
Simulate annual benefit receipt given household size, household income, & year
Benefits = max_benefit - (reduction_param * monthly_income) + elderly bonus
CPI used to scale benefits to present (base year is 2023)
'''
def simulate_benefit(hh, inc, r65, yr, prog):
    if prog == 'NAP':
      max_benefit = NAPmax_benefit_schedule[hh]
      max_income = NAPmax_eligincome_schedule[hh]
    else:
      max_benefit = SNAPmax_benefit_schedule[hh]
      max_income = SNAPmax_eligincome_schedule[hh]

    if r65 > 0:
      elder = 1000
    else:
      elder = 0

    if (inc / 12) > max_income:
      benefits =  0
    else:
      benefits = (max(0, (max_benefit - (0.3 * (inc / 12)))) * 12) + elder

    return (benefits * (CPI[yr] / CPI[2024]))

df['R65'] = df['R65'].astype(int)
vect = np.vectorize(simulate_benefit)

df['SIMBENEFIT'] = vect(df['HHSIZE'], df['avg_HINCP'], df['R65'], 2024, 'NAP')
df['SQ_SIMBENEFIT'] = np.square(df['SIMBENEFIT'])
df['SIMBENEFIT_SNAP'] = vect(df['HHSIZE'], df['avg_HINCP'], df['R65'], 2024, 'SNAP')

#interact simbenefits with edu
df['SB_EDU2'] = df['SIMBENEFIT'] * df['EDU2']
df['SB_EDU3'] = df['SIMBENEFIT'] * df['EDU3']
df['SB_EDU4'] = df['SIMBENEFIT'] * df['EDU4']
df['SB_EDU5'] = df['SIMBENEFIT'] * df['EDU5']


## [MEN'S EMPLOYMENT]

df['SEX'] = df['SEX'].astype(int)
df_men = df[df['SEX'] == 1]

df_men = df_men.dropna(subset = ['ESR', 'ADJ_HINCP', 'SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'AGEP', 'MARRIED', 'SQ_AGEP']).copy()

Xs = stats.add_constant(df_men[['SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'unemp_reg', 'AGEP', 'MARRIED', 'SQ_AGEP']])
Ys = df_men['ESR']
weights = df_men['PWGTP'].astype(float)

SE = stats.GLM(Ys, Xs, family=Binomial(link=Probit()), freq_weights=weights)
probit_outcome = SE.fit()

print(probit_outcome.summary())

me = probit_outcome.get_margeff(at='overall')
print(me.summary())

#calculate extensive margin
beta1 = probit_outcome.params['SIMBENEFIT']
xi = probit_outcome.predict(Xs, which= "linear")
phi = norm.pdf(xi)
Phi = norm.cdf(xi)
epsilone = (phi * beta1 * df_men['SIMBENEFIT'])
df_men['epsilon'] = epsilone

#average elasticity (weighted)
epsilon_bar = np.average(df_men['epsilon'], weights=weights)

#put predicted employment into df_men
xi_NAP = probit_outcome.predict(Xs, which="linear")
df_men['P_EMP_NAP'] = norm.cdf(xi_NAP)

#counterfactual (SNAP)
Xs_SNAP = Xs.copy()
Xs_SNAP['SIMBENEFIT'] = df_men['SIMBENEFIT_SNAP']

xi_SNAP = probit_outcome.predict(Xs_SNAP, which="linear")
df_men['P_EMP_SNAP'] = norm.cdf(xi_SNAP)


## [WOMEN'S EMPLOYMENT]

df['SEX'] = df['SEX'].astype(int)
df_women = df[df['SEX'] == 0]

df_women = df_women.dropna(subset = ['ESR', 'ADJ_HINCP', 'SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'AGEP', 'SQ_AGEP', 'MARRIED']).copy()

Xs = stats.add_constant(df_women[['SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'unemp_reg', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'NOC_MAR', 'OCunder6_MAR', 'AGEP', 'SQ_AGEP', 'MARRIED']])
Ys = df_women['ESR']
weights = df_women['PWGTP'].astype(float)

SE = stats.GLM(Ys, Xs, family=Binomial(link=Probit()), freq_weights=weights)
probit_outcome = SE.fit()

print(probit_outcome.summary())

me = probit_outcome.get_margeff(at='overall')
print(me.summary())

#calculate extensive margin
beta1 = probit_outcome.params['SIMBENEFIT']
xi = probit_outcome.predict(Xs, which= "linear")
phi = norm.pdf(xi)
Phi = norm.cdf(xi)
epsilone = (phi * beta1 * df_women['SIMBENEFIT'])
df_women['epsilon'] = epsilone

#average elasticity (weighted)
epsilon_bar = np.average(df_women['epsilon'], weights=weights)

#put predicted employment into df_women
xi_NAP = probit_outcome.predict(Xs, which="linear")
df_women['P_EMP_NAP'] = norm.cdf(xi_NAP)

#counterfactual (SNAP)
Xs_SNAP = Xs.copy()
Xs_SNAP['SIMBENEFIT'] = df_women['SIMBENEFIT_SNAP']

xi_SNAP = probit_outcome.predict(Xs_SNAP, which="linear")
df_women['P_EMP_SNAP'] = norm.cdf(xi_SNAP)


## [OVERALL EMPLOYMENT IMPACTS]

#combine men and women
df_comb = pd.concat([df_women, df_men], axis = 0)
weights = df_comb['PWGTP'].astype(int)

#average elasticity (weighted)
epsilon_bar = np.average(df_comb['epsilon'], weights=weights)

print(epsilon_bar)

df_comb['P_UNEMP_NAP'] = 1 - df_comb['P_EMP_NAP']

#change
df_comb['EMP_change'] = df_comb['P_EMP_SNAP'] - df_comb['P_EMP_NAP']
df_comb['EMP_change_pct'] = df_comb['EMP_change'] / df_comb['P_EMP_NAP'] *100

#average
avg = np.average(df_comb['EMP_change_pct'], weights=weights)
totalchange = np.sum(df_comb['EMP_change'] * weights)

baseline_employment = np.sum(df_comb['P_EMP_NAP'] * weights)
baseline_unemp = np.sum(df_comb['P_UNEMP_NAP'] * weights)
pct_change_total = (totalchange / baseline_employment) * 100

print(avg)
print(totalchange)
print(baseline_employment)
print(baseline_unemp)
print(pct_change_total)


## [MEN'S HOURS WORKED]

df['SEX'] = df['SEX'].astype(int)
df['WKHP'] = df['WKHP'].astype(int)

df_men = df[df['SEX'] == 1]
df_men =df_men[df_men['WKHP'] > 0].copy()
df_men = df_men[~df_men['INDP_code'].isin([12, 19])].copy()
df_men = df_men.dropna(subset = ['ADJ_HINCP']).copy()

Xo = stats.add_constant(df_men[['SIMBENEFIT', 'ADJ_WAGP' , 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'AGEP', 'SQ_AGEP', 'IND2', 'IND3', 'IND4', 'IND5', 'IND6', 'MARRIED']])
Yo = df_men['WKHP']
weights = df_men['PWGTP'].astype(float)

OE = stats.WLS(Yo, Xo, weights=weights)
outcome = OE.fit()

print(outcome.summary())

#elasticity
beta1 = outcome.params['SIMBENEFIT']
elast = beta1 * (df_men['SIMBENEFIT'] / df_men['WKHP'])
df_men['hrs_elast'] = elast

#average elasticity
epsilon_bar = np.average(df_men['hrs_elast'], weights = weights)

#put predicted hours into df_men
df_men['WKHP_NAP'] = outcome.predict(Xo)

#counterfactual (SNAP)
Xo_SNAP = Xo.copy()
Xo_SNAP['SIMBENEFIT'] = df_men['SIMBENEFIT_SNAP']
df_men['WKHP_SNAP'] = outcome.predict(Xo_SNAP)



## [WOMEN'S HOURS WORKED]

df['SEX'] = df['SEX'].astype(int)
df['WKHP'] = df['WKHP'].astype(int)

df_women = df[df['SEX'] == 0]
df_women =df_women[df_women['WKHP'] > 0].copy()
df_women = df_women[~df_women['INDP_code'].isin([12, 19])].copy()
df_women = df_women.dropna(subset = ['ADJ_WAGP', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'NOC', 'OC_under6', 'FER', 'AGEP', 'SQ_AGEP']).copy()

Xo = stats.add_constant(df_women[['SIMBENEFIT', 'ADJ_WAGP' ,'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE','SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'AGEP', 'SQ_AGEP', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'NOC_MAR', 'IND2', 'IND3', 'IND4', 'IND5', 'IND6', 'MARRIED']])
Yo = df_women['WKHP']
weights = df_women['PWGTP'].astype(float)

OE = stats.WLS(Yo, Xo, weights=weights)
outcome = OE.fit()

print(outcome.summary())

#elasticity
beta1 = outcome.params['SIMBENEFIT']
elast = beta1 * (df_women['SIMBENEFIT'] / df_women['WKHP'])
df_women['hrs_elast'] = elast

#average elasticity
epsilon_bar = np.average(df_women['hrs_elast'], weights = weights)

#put predicted hours into df_women
df_women['WKHP_NAP'] = outcome.predict(Xo)

#counterfactual (SNAP)
Xo_SNAP = Xo.copy()
Xo_SNAP['SIMBENEFIT'] = df_women['SIMBENEFIT_SNAP']
df_women['WKHP_SNAP'] = outcome.predict(Xo_SNAP)


## [OVERALL HOURS WORKED]
#combine men and women
df_comb = pd.concat([df_women, df_men], axis = 0)

#df_comb = df_comb[np.isfinite(df_comb['hrs_elast'])]
weights = df_comb['PWGTP'].astype(int)

#average elasticity (weighted)
epsilon_bar = np.average(df_comb['hrs_elast'], weights=weights)

print(epsilon_bar)

#change
df_comb['WKHP_change'] = df_comb['WKHP_SNAP'] - df_comb['WKHP_NAP']
df_comb['WKHP_change_pct'] = df_comb['WKHP_change'] / df_comb['WKHP_NAP'] * 100

#average
avg = np.average(df_comb['WKHP_change_pct'], weights=weights)
totalchange = np.sum(df_comb['WKHP_change'] * weights)

baseline_wkhp = np.sum(df_comb['WKHP_NAP'] * weights)
pct_change_total = (totalchange / baseline_wkhp) * 100

print(avg)
print(totalchange)
print(baseline_wkhp)
print(pct_change_total)


## [MEN'S TAKE UP]

df['SEX'] = df['SEX'].astype(int)
df_men = df[df['SEX'] == 1].copy()

df_men['Elig_NAP'] = (
    (df_men['ADJ_HINCP'] / 12 ) <= df_men['HHSIZE'].map(NAPmax_eligincome_schedule)
).astype(int)

df_men['Elig_SNAP'] = (
    (df_men['ADJ_HINCP'] / 12 ) <= df_men['HHSIZE'].map(SNAPmax_eligincome_schedule)
).astype(int)

df_men_elig = df_men[(df_men['Elig_NAP'] == 1) & (df_men['Elig_SNAP'] == 1)].copy()

df_men_elig['SIMBENEFIT'] = df_men_elig['SIMBENEFIT'].astype(int)
df_men_elig['HHSIZE'] = df_men_elig['HHSIZE'].astype(int)

X4 = stats.add_constant(df_men_elig[['SIMBENEFIT', 'HHSIZE', 'R65', 'AGEP', 'SQ_AGEP', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'MARRIED', 'fs_availibility', 'fs_lagreceipt']])
Y4 = df_men_elig['FS']
weights = df_men_elig['WGTP'].astype(float)

S4 = stats.GLM(Y4, X4, family=Binomial(link=Probit()), freq_weights=weights).fit()
print(S4.summary())

me = S4.get_margeff(at='overall')
print(me.summary())

#calculate extensive margin
beta1 = S4.params['SIMBENEFIT']
xi = S4.predict(X4, which= "linear")
phi = norm.pdf(xi)
Phi = norm.cdf(xi)
epsilone = (phi * beta1* df_men_elig['SIMBENEFIT']) / Phi
df_men_elig['epsilon'] = epsilone

#get predicted values for NAP
df_men_elig['P_takeup_NAP'] = Phi
df_men_elig['P_nontake_NAP'] = 1 - df_men_elig['P_takeup_NAP']

#do counterfactual
X4_SNAP = X4.copy()
X4_SNAP['SIMBENEFIT'] = df_men_elig['SIMBENEFIT_SNAP']

xi_SNAP = S4.predict(X4_SNAP, which="linear")
df_men_elig['P_takeup_SNAP'] = norm.cdf(xi_SNAP)
df_men_elig['P_nontake_SNAP'] = 1 - df_men_elig['P_takeup_SNAP']


## [WOMEN'S TAKE UP]

df['SEX'] = df['SEX'].astype(int)
df_women = df[df['SEX'] == 0].copy()

df_women['Elig_NAP'] = (
    (df_women['ADJ_HINCP'] / 12 ) <= df_women['HHSIZE'].map(NAPmax_eligincome_schedule)
).astype(int)

df_women['Elig_SNAP'] = (
    (df_women['ADJ_HINCP'] / 12 ) <= df_women['HHSIZE'].map(SNAPmax_eligincome_schedule)
).astype(int)

df_women_elig = df_women[(df_women['Elig_NAP'] == 1) & (df_women['Elig_SNAP'] == 1)].copy()

df_women_elig['SIMBENEFIT'] = df_women_elig['SIMBENEFIT'].astype(int)
df_women_elig['HHSIZE'] = df_women_elig['HHSIZE'].astype(int)

df_women_elig = df_women_elig.dropna(subset = ['EDU2', 'EDU3', 'EDU4' , 'EDU5', 'NOC', 'OC_under6', 'FER', 'AGEP', 'SQ_AGEP']).copy()

X4 = stats.add_constant(df_women_elig[['SIMBENEFIT', 'HHSIZE', 'R65', 'AGEP', 'SQ_AGEP', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'NOC_MAR', 'OCunder6_MAR', 'MARRIED', 'fs_availibility', 'fs_lagreceipt']])
Y4 = df_women_elig['FS']
weights = df_women_elig['WGTP'].astype(float)

S4 = stats.GLM(Y4, X4, family=Binomial(link=Probit()), freq_weights=weights).fit()
print(S4.summary())

me = S4.get_margeff(at='overall')
print(me.summary())

#calculate extensive margin
beta1 = S4.params['SIMBENEFIT']
xi = S4.predict(X4, which= "linear")
phi = norm.pdf(xi)
Phi = norm.cdf(xi)
epsilone = (phi * beta1* df_women_elig['SIMBENEFIT']) / Phi
df_women_elig['epsilon'] = epsilone

#get predicted values for NAP
df_women_elig['P_takeup_NAP'] = Phi
df_women_elig['P_nontake_NAP'] = 1 - df_women_elig['P_takeup_NAP']

#do counterfactual
X4_SNAP = X4.copy()
X4_SNAP['SIMBENEFIT'] = df_women_elig['SIMBENEFIT_SNAP']

xi_SNAP = S4.predict(X4_SNAP, which="linear")
df_women_elig['P_takeup_SNAP'] = norm.cdf(xi_SNAP)
df_women_elig['P_nontake_SNAP'] = 1 - df_women_elig['P_takeup_SNAP']

## [OVERALL TAKE UP]

#combine men and women
df_comb = pd.concat([df_women_elig, df_men_elig], axis = 0)
weights = df_comb['PWGTP'].astype(int)

#average elasticity (weighted)
epsilon_bar = np.average(df_comb['epsilon'], weights=weights)

print(epsilon_bar)

#change
df_comb['takeup_change'] = df_comb['P_takeup_SNAP'] - df_comb['P_takeup_NAP']
df_comb['takeup_change_pct'] = df_comb['takeup_change'] / df_comb['P_takeup_NAP'] *100

#average
avg = np.average(df_comb['takeup_change_pct'], weights=weights)
totalchange = np.sum(df_comb['takeup_change'] * weights)

baseline_takeup = np.sum(df_comb['P_takeup_NAP'] * weights)
baseline_nontake = np.sum(df_comb['P_nontake_NAP'] * weights)
pct_change_total = (totalchange / baseline_takeup) * 100

counter_take = np.sum(df_comb['P_takeup_SNAP'] * weights)
counter_nontake = np.sum(df_comb['P_nontake_SNAP'] * weights)

print(avg)
print(totalchange)
print(baseline_takeup)
print(baseline_nontake)
print(pct_change_total)

print(counter_take)
print(counter_nontake)
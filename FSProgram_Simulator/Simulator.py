## [LOAD PACKAGES]
import pandas as pd
import requests
import numpy as np
import statsmodels.api as stats
from scipy.stats import norm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Probit


## [SET UP SOME VARIABLES]

## state codes
state_codes = {
	"AL": "01",
	"AK": "02",
	"AZ": "04",
	"AR": "05",
	"CA": "06",
	"CO": "08",
	"CT": "09",
	"DE": "10",
	"DC": "11",
	"FL": "12",
	"GA": "13",
	"HI": "15",
	"ID": "16",
	"IL": "17",
	"IN": "18",
	"IA": "19",
	"KS": "20",
	"KY": "21",
	"LA": "22",
	"ME": "23",
	"MD": "24",
	"MA": "25",
	"MI": "26",
	"MN": "27",
	"MS": "28",
	"MO": "29",
	"MT": "30",
	"NE": "31",
	"NV": "32",
	"NH": "33",
	"NJ": "34",
	"NM": "35",
	"NY": "36",
	"NC": "37",
	"ND": "38",
	"OH": "39",
	"OK": "40",
	"OR": "41",
	"PA": "42",
	"RI": "44",
	"SC": "45",
	"SD": "46",
	"TN": "47",
	"TX": "48",
	"UT": "49",
	"VT": "50",
	"VA": "51",
	"WA": "53",
	"WV": "54",
	"WI": "55",
	"WY": "56",
	"PR": "72"
}

## NAP benefits
NAPmax_benefit_schedule = {1:183, 2:337, 3:482, 4:612, 5:727, 6:873, 7:965,
							8:1103, 9:1241, 10:1379, 11:1517, 12:1655,
							13:1792, 14:1930, 15:2068, 16:2206, 17:2344, 18:2482}
NAPmax_eligincome_schedule = {1:619, 2:1179, 3:1706, 4:2206, 5:2666, 6:3199, 7:3599,
							8:4112, 9:4632, 10:5152, 11:5666, 12:6186,
							13:6699, 14:7219, 15:7732, 16:8252, 17:8766, 18:9286}


## SNAP benefits
SNAPmax_benefit_schedule = {1:291, 2:535, 3:766, 4:973, 5:1155, 6:1386,
                            7:1532, 8:1751, 9:1970, 10:2189, 11:2408, 12:2627,
                            13:2846, 14:3065, 15:3284, 16:3503, 17:3722, 18:3941}

SNAPmax_eligincome_schedule = {1:1215, 2:1644, 3:2072, 4:2500, 5:2929, 6:3357,
                               7:3785, 8:4214, 9:4643, 10:5072, 11:5501, 12:5930,
                               13:6359, 14:6788, 15:7217, 16:7646, 17:8075, 18:8504}





## [FETCHING DATA]

print("\n-- CONSTRUCT YOUR FOOD STAMP PROGRAM SCHEDULE--")

while True:
	yr = input("What year is the simulation for? Enter either 2023 or 2024 as a number with no extra spaces: \n")

	if len(yr) != 4:
		print("You must choose either 2023 or 2024.")
		continue

	try:
		yr = int(yr)
	except ValueError:
		print("Input must be an year (number)!")
		continue

	if yr not in (2023, 2024):
		print("You must choose either 2023 or 2024.")
		continue

	break


while True:
	code = input("\nWhat location is the simulation for? Enter the two letter abbreviation with no spaces for one of the U.S. states or PR for Puerto Rico:\n")

	if len(code) != 2:
		print("You must use the two letter abbreviation of a state.")
		continue

	if code not in ("AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID",
				 	"IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO",
					"MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA",
					"RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","PR"):
		print("You must choose a U.S. state.")
		continue

	break

user_state = state_codes[code]


if user_state == "72":
	df = pd.read_csv(f"{yr}Dataset.csv")
else:
	while True:
		API_key = input("\nPlease enter your US Census API key:\n")

		if len(API_key) < 10:
			print("You must enter a valid API key for your simulation to work")
			continue

		break

	url = f"https://api.census.gov/data/{yr}/acs/acs1/pums"

	vars = {
		"get": "PWGTP,WGTP,AGEP,SEX,MAR,SCHL,WKHP,WKWN,WAGP,HINCP,INDP,ESR,FS,SERIALNO,R65,NOC,NRC,FER,PUMA,ADJINC,HUPAOC,HUPARC",
		"for": f"state:{user_state}",
		"key" : API_key
	}

	print("\nFetching data for your state, this may take some time ...\n")

	variables = requests.get(url, params=vars)
	if variables.status_code != 200:
		raise ValueError(f"There was an issue processing the US Census API request with code {variables.status_code}.")

	arrays = variables.json()
	df = pd.DataFrame(arrays[1:], columns=arrays[0])


	## [SETTING UP DATASET]

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

	if yr == 2023:
		indp_json = pd.read_json("INDP.json")
	else:
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

	#create Avg_HHIncome
	df['PWGTP'] = df['PWGTP'].astype(int)
	df_inc = df.dropna(subset = 'ADJ_HINCP')
	vals = df_inc.groupby(['PUMA', 'SEX', 'INDP_code']).apply(lambda group: np.average(group['ADJ_WAGP'], weights=group['PWGTP'])).reset_index(name='mean_WAGP')
	#print(vals)

	vals['PUMA'] = vals['PUMA'].astype('Int64')
	df['PUMA'] = df['PUMA'].astype('Int64')

	mapping = dict(zip(zip(vals['PUMA'], vals['SEX'], vals['INDP_code']), vals['mean_WAGP']))

	df['avg_WAGP'] = pd.Series(list(zip(df['PUMA'], df['SEX'], df['INDP_code']))).map(mapping)

	df['avg_HINCP'] = df.groupby('SERIALNO')['avg_WAGP'].transform('sum')






## [SET UP SIMULATED BENEFITS]

while True:
	values = (input("\nEnter values for maximum benefit amount allowable to each household size in accesinding order separated by commas (no spaces): \n \n Ex: 291,535,766,973,1155,1386,1532,1751,1970,2189,2408,2627,2846,3065,3284,3503,3722,3941 \n The above example corresponds to the SNAP maximum benefit schedule \n").split(","))

	if len(values) != 18:
		print("You must enter 18 values.")
		continue

	try:
		values = list(map(int, values))
		break
	except ValueError:
		print("All values must be numbers (integers).")

NEWmax_benefit_schedule = {i + 1: values[i] for i in range(18)}

print("  ")

while True:
	values = (input("\nEnter values for income cutoff for each household size to be eligible in accesinding order separated by commas (no spaces): \n \n Ex: 1215,1644,2072,2500,2929,3357,3785,4214,4643,5072,5501,5930,6359,6788,7217,7646,8075,8504 \n The above example corresponds to the SNAP income eligibility cutoff schedule \n").split(","))

	if len(values) != 18:
		print("You must enter 18 values.")
		continue

	try:
		values = list(map(int, values))
		break
	except ValueError:
		print("All values must be numbers (integers).")

NEWmax_eligincome_schedule = {i + 1: values[i] for i in range(18)}

print("  ")


#CPI values by year
CPI = {2023:304.702, 2024:313.689}

'''
Simulate annual benefit receipt given household size, household income, & year
Benefits = max_benefit - (reduction_param * monthly_income) + elderly bonus
CPI used to scale benefits to present (base year is 2023)
'''
def simulate_benefit(hh, inc, r65, yr, prog):
	if prog == "NAP":
		max_benefit = NAPmax_benefit_schedule[hh]
		max_income = NAPmax_eligincome_schedule[hh]
	elif prog == "SNAP":
		max_benefit = SNAPmax_benefit_schedule[hh]
		max_income = SNAPmax_eligincome_schedule[hh]
	else:
		max_benefit = NEWmax_benefit_schedule[hh]
		max_income = NEWmax_eligincome_schedule[hh]

	if r65 > 0:
		elder = 1000
	else:
		elder = 0

	if (inc / 12) > max_income:
		benefits =  0
	else:
		benefits = (max(0, (max_benefit - (0.3 * (inc / 12)))) * 12) + elder

	return (benefits * (CPI[yr] / CPI[2024]))




if user_state == "72":
	baseline = "NAP"
else:
	baseline = "SNAP"

df['R65'] = df['R65'].astype(int)
df = df[df['HHSIZE'] <= 18]
vect = np.vectorize(simulate_benefit)

df['SIMBENEFIT'] = vect(df['HHSIZE'], df['avg_HINCP'], df['R65'], yr, baseline)
df['SIMBENEFIT_NEW'] = vect(df['HHSIZE'], df['avg_HINCP'], df['R65'], yr, 'NEW')

#interact simbenefits with edu
df['SB_EDU2'] = df['SIMBENEFIT'] * df['EDU2']
df['SB_EDU3'] = df['SIMBENEFIT'] * df['EDU3']
df['SB_EDU4'] = df['SIMBENEFIT'] * df['EDU4']
df['SB_EDU5'] = df['SIMBENEFIT'] * df['EDU5']



print("\n Calculating the estimated impacts, this may take some time ...\n")



## [EMPLOYMENT]

# men's
df['SEX'] = df['SEX'].astype(int)
df_men = df[df['SEX'] == 1]

df_men = df_men.dropna(subset = ['ESR', 'ADJ_HINCP', 'SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'AGEP', 'MARRIED', 'SQ_AGEP']).copy()

if user_state == "72":
	Xs = stats.add_constant(df_men[['SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'unemp_reg', 'AGEP', 'MARRIED', 'SQ_AGEP']])
else:
	Xs = stats.add_constant(df_men[['SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'AGEP', 'MARRIED', 'SQ_AGEP']])

Ys = df_men['ESR']
weights = df_men['PWGTP'].astype(float)

SE = stats.GLM(Ys, Xs, family=Binomial(link=Probit()), freq_weights=weights)
probit_outcome = SE.fit()
m_emp_simben = [probit_outcome.pvalues['SIMBENEFIT'], "men"]

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
xi_baseline = probit_outcome.predict(Xs, which="linear")
df_men['P_EMP_baseline'] = norm.cdf(xi_baseline)

#counterfactual
Xs_NEW = Xs.copy()
Xs_NEW['SIMBENEFIT'] = df_men['SIMBENEFIT_NEW']

xi_NEW = probit_outcome.predict(Xs_NEW, which="linear")
df_men['P_EMP_NEW'] = norm.cdf(xi_NEW)


#women's
df['SEX'] = df['SEX'].astype(int)
df_women = df[df['SEX'] == 0]

df_women = df_women.dropna(subset = ['ESR', 'ADJ_HINCP', 'SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'AGEP', 'SQ_AGEP', 'MARRIED']).copy()

if user_state == "72":
	Xs = stats.add_constant(df_women[['SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'unemp_reg', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'NOC_MAR', 'OCunder6_MAR', 'AGEP', 'SQ_AGEP', 'MARRIED']])
else:
	Xs = stats.add_constant(df_women[['SIMBENEFIT', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'EDU_AGE', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'NOC_MAR', 'OCunder6_MAR', 'AGEP', 'SQ_AGEP', 'MARRIED']])

Ys = df_women['ESR']
weights = df_women['PWGTP'].astype(float)

SE = stats.GLM(Ys, Xs, family=Binomial(link=Probit()), freq_weights=weights)
probit_outcome = SE.fit()
w_emp_simben = [probit_outcome.pvalues['SIMBENEFIT'], "women"]

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
xi_baseline = probit_outcome.predict(Xs, which="linear")
df_women['P_EMP_baseline'] = norm.cdf(xi_baseline)

#counterfactual
Xs_NEW = Xs.copy()
Xs_NEW['SIMBENEFIT'] = df_women['SIMBENEFIT_NEW']

xi_NEW = probit_outcome.predict(Xs_NEW, which="linear")
df_women['P_EMP_NEW'] = norm.cdf(xi_NEW)


#combine men and women
df_comb = pd.concat([df_women, df_men], axis = 0)
weights = df_comb['PWGTP'].astype(int)

#average elasticity (weighted)
epsilon_bar = np.average(df_comb['epsilon'], weights=weights)

df_comb['P_UNEMP_baseline'] = 1 - df_comb['P_EMP_baseline']

#change
df_comb['EMP_change'] = df_comb['P_EMP_NEW'] - df_comb['P_EMP_baseline']
df_comb['EMP_change_pct'] = df_comb['EMP_change'] / df_comb['P_EMP_baseline'] *100

#average
avg = np.average(df_comb['EMP_change_pct'], weights=weights)
totalchange = np.sum(df_comb['EMP_change'] * weights)

baseline_employment = np.sum(df_comb['P_EMP_baseline'] * weights)
baseline_unemp = np.sum(df_comb['P_UNEMP_baseline'] * weights)
pct_change_total = (totalchange / baseline_employment) * 100


print("-- EMPLOYMENT --")

for i in [m_emp_simben, w_emp_simben]:
	if i[0] < 0.01:
		sig = "1%"
	elif i[0] < 0.05:
		sig = "5%"
	elif i[0] < 0.10:
		sig = "10%"
	else:
		sig = "N/A"
	print(f"Coefficient on SimBenefit is signficant at {sig} for {i[1]}")

print("Average Employment Elasticity: ", round(epsilon_bar, 5))
print("Average Individual Employment Change: ", round(avg, 5))
print("Overall Change in Employment: ", round(pct_change_total, 5))
print("Total Employed Individuals Under Baseline: ", round(baseline_employment, 5))
print("Total Employed Individuals Under New: ", round(baseline_employment + totalchange, 5))

print("   ")

## [WEEKLY HOURS WORKED]

# men's 
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
m_hrs_simben = [outcome.pvalues['SIMBENEFIT'], "men"]

#elasticity
beta1 = outcome.params['SIMBENEFIT']
elast = beta1 * (df_men['SIMBENEFIT'] / df_men['WKHP'])
df_men['hrs_elast'] = elast

#average elasticity
epsilon_bar = np.average(df_men['hrs_elast'], weights = weights)

#put predicted hours into df_men
df_men['WKHP_baseline'] = outcome.predict(Xo)

#counterfactual
Xo_NEW = Xo.copy()
Xo_NEW['SIMBENEFIT'] = df_men['SIMBENEFIT_NEW']
df_men['WKHP_NEW'] = outcome.predict(Xo_NEW)


#women's

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
w_hrs_simben = [outcome.pvalues['SIMBENEFIT'], "women"]

#elasticity
beta1 = outcome.params['SIMBENEFIT']
elast = beta1 * (df_women['SIMBENEFIT'] / df_women['WKHP'])
df_women['hrs_elast'] = elast

#average elasticity
epsilon_bar = np.average(df_women['hrs_elast'], weights = weights)

#put predicted hours into df_women
df_women['WKHP_baseline'] = outcome.predict(Xo)

#counterfactual
Xo_NEW = Xo.copy()
Xo_NEW['SIMBENEFIT'] = df_women['SIMBENEFIT_NEW']
df_women['WKHP_NEW'] = outcome.predict(Xo_NEW)


#combine men and women
df_comb = pd.concat([df_women, df_men], axis = 0)

#df_comb = df_comb[np.isfinite(df_comb['hrs_elast'])]
weights = df_comb['PWGTP'].astype(int)

#average elasticity (weighted)
epsilon_bar = np.average(df_comb['hrs_elast'], weights=weights)

#change
df_comb['WKHP_change'] = df_comb['WKHP_NEW'] - df_comb['WKHP_baseline']
df_comb['WKHP_change_pct'] = df_comb['WKHP_change'] / df_comb['WKHP_baseline'] * 100

#average
avg = np.average(df_comb['WKHP_change_pct'], weights=weights)
totalchange = np.sum(df_comb['WKHP_change'] * weights)

baseline_wkhp = np.sum(df_comb['WKHP_baseline'] * weights)
pct_change_total = (totalchange / baseline_wkhp) * 100


print("-- WEEKLY HOURS WORKED --")

for i in [m_hrs_simben, w_hrs_simben]:
	if i[0] < 0.01:
		sig = "1%"
	elif i[0] < 0.05:
		sig = "5%"
	elif i[0] < 0.10:
		sig = "10%"
	else:
		sig = "N/A"
	print(f"Coefficient on SimBenefit is signficant at {sig} for {i[1]}")

print("Average Hours Worked Elasticity: ", round(epsilon_bar, 5))
print("Average Individual Change in Weekly Hours Worked: ", round(avg, 5))
print("Overall Change in Weekly Hours Worked: ", round(pct_change_total, 5))
print("Total Weekly Hours Worked Under Baseline: ", round(baseline_wkhp, 5))
print("Total Weekly Hours Under New: ", round(baseline_wkhp + totalchange, 5))

print("   ")



## [TAKE UP]

#men's
df['SEX'] = df['SEX'].astype(int)
df_men = df[df['SEX'] == 1].copy()

if user_state == "72":
	df_men['Elig_baseline'] = (
    	(df_men['ADJ_HINCP'] / 12 ) <= df_men['HHSIZE'].map(NAPmax_eligincome_schedule)
	).astype(int)
else:
	df_men['Elig_baseline'] = (
    	(df_men['ADJ_HINCP'] / 12 ) <= df_men['HHSIZE'].map(SNAPmax_eligincome_schedule)
	).astype(int)

df_men['Elig_NEW'] = (
    (df_men['ADJ_HINCP'] / 12 ) <= df_men['HHSIZE'].map(NEWmax_eligincome_schedule)
).astype(int)

df_men_elig = df_men[(df_men['Elig_baseline'] == 1) & (df_men['Elig_NEW'] == 1)].copy()

df_men_elig['SIMBENEFIT'] = df_men_elig['SIMBENEFIT'].astype(int)
df_men_elig['HHSIZE'] = df_men_elig['HHSIZE'].astype(int)

if user_state == "72":
	X4 = stats.add_constant(df_men_elig[['SIMBENEFIT', 'HHSIZE', 'R65', 'AGEP', 'SQ_AGEP', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'MARRIED', 'fs_availibility', 'fs_lagreceipt']])
else:
	X4 = stats.add_constant(df_men_elig[['SIMBENEFIT', 'HHSIZE', 'R65', 'AGEP', 'SQ_AGEP', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'MARRIED']])

Y4 = df_men_elig['FS']
weights = df_men_elig['WGTP'].astype(float)

S4 = stats.GLM(Y4, X4, family=Binomial(link=Probit()), freq_weights=weights).fit()
m_ben_simben = [S4.pvalues['SIMBENEFIT'], "men"]

#calculate extensive margin
beta1 = S4.params['SIMBENEFIT']
xi = S4.predict(X4, which= "linear")
phi = norm.pdf(xi)
Phi = norm.cdf(xi)
epsilone = (phi * beta1* df_men_elig['SIMBENEFIT']) / Phi
df_men_elig['epsilon'] = epsilone

#get predicted values for baseline
df_men_elig['P_takeup_baseline'] = Phi
df_men_elig['P_nontake_baseline'] = 1 - df_men_elig['P_takeup_baseline']

#do counterfactual
X4_NEW = X4.copy()
X4_NEW['SIMBENEFIT'] = df_men_elig['SIMBENEFIT_NEW']

xi_NEW = S4.predict(X4_NEW, which="linear")
df_men_elig['P_takeup_NEW'] = norm.cdf(xi_NEW)
df_men_elig['P_nontake_NEW'] = 1 - df_men_elig['P_takeup_NEW']


#women's
df['SEX'] = df['SEX'].astype(int)
df_women = df[df['SEX'] == 0].copy()

if user_state == "72":
	df_women['Elig_baseline'] = (
    	(df_women['ADJ_HINCP'] / 12 ) <= df_women['HHSIZE'].map(NAPmax_eligincome_schedule)
	).astype(int)
else:
	df_women['Elig_baseline'] = (
		(df_women['ADJ_HINCP'] / 12 ) <= df_women['HHSIZE'].map(SNAPmax_eligincome_schedule)
	).astype(int)

df_women['Elig_NEW'] = (
    (df_women['ADJ_HINCP'] / 12 ) <= df_women['HHSIZE'].map(NEWmax_eligincome_schedule)
).astype(int)

df_women_elig = df_women[(df_women['Elig_baseline'] == 1) & (df_women['Elig_NEW'] == 1)].copy()

df_women_elig['SIMBENEFIT'] = df_women_elig['SIMBENEFIT'].astype(int)
df_women_elig['HHSIZE'] = df_women_elig['HHSIZE'].astype(int)

df_women_elig = df_women_elig.dropna(subset = ['EDU2', 'EDU3', 'EDU4' , 'EDU5', 'NOC', 'OC_under6', 'FER', 'AGEP', 'SQ_AGEP']).copy()

if user_state == "72":
	X4 = stats.add_constant(df_women_elig[['SIMBENEFIT', 'HHSIZE', 'R65', 'AGEP', 'SQ_AGEP', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'NOC_MAR', 'OCunder6_MAR', 'MARRIED', 'fs_availibility', 'fs_lagreceipt']])
else:
	X4 = stats.add_constant(df_women_elig[['SIMBENEFIT', 'HHSIZE', 'R65', 'AGEP', 'SQ_AGEP', 'EDU2', 'EDU3', 'EDU4' , 'EDU5', 'SB_EDU2', 'SB_EDU3', 'SB_EDU4', 'SB_EDU5', 'NOC', 'OC_under6', 'FER', 'FER_MAR', 'NOC_MAR', 'OCunder6_MAR', 'MARRIED']])

Y4 = df_women_elig['FS']
weights = df_women_elig['WGTP'].astype(float)

S4 = stats.GLM(Y4, X4, family=Binomial(link=Probit()), freq_weights=weights).fit()
w_ben_simben = [S4.pvalues['SIMBENEFIT'], "women"]

#calculate extensive margin
beta1 = S4.params['SIMBENEFIT']
xi = S4.predict(X4, which= "linear")
phi = norm.pdf(xi)
Phi = norm.cdf(xi)
epsilone = (phi * beta1* df_women_elig['SIMBENEFIT']) / Phi
df_women_elig['epsilon'] = epsilone

#get predicted values for baseline
df_women_elig['P_takeup_baseline'] = Phi
df_women_elig['P_nontake_baseline'] = 1 - df_women_elig['P_takeup_baseline']

#do counterfactual
X4_NEW = X4.copy()
X4_NEW['SIMBENEFIT'] = df_women_elig['SIMBENEFIT_NEW']

xi_NEW = S4.predict(X4_NEW, which="linear")
df_women_elig['P_takeup_NEW'] = norm.cdf(xi_NEW)
df_women_elig['P_nontake_NEW'] = 1 - df_women_elig['P_takeup_NEW']


#combine men and women
df_comb = pd.concat([df_women_elig, df_men_elig], axis = 0)
weights = df_comb['PWGTP'].astype(int)

#average elasticity (weighted)
epsilon_bar = np.average(df_comb['epsilon'], weights=weights)

#change
df_comb['takeup_change'] = df_comb['P_takeup_NEW'] - df_comb['P_takeup_baseline']
df_comb['takeup_change_pct'] = df_comb['takeup_change'] / df_comb['P_takeup_baseline'] *100

#average
avg = np.average(df_comb['takeup_change_pct'], weights=weights)
totalchange = np.sum(df_comb['takeup_change'] * weights)

baseline_takeup = np.sum(df_comb['P_takeup_baseline'] * weights)
baseline_nontake = np.sum(df_comb['P_nontake_baseline'] * weights)
pct_change_total = (totalchange / baseline_takeup) * 100

counter_take = np.sum(df_comb['P_takeup_NEW'] * weights)
counter_nontake = np.sum(df_comb['P_nontake_NEW'] * weights)


print("-- BENEFIT TAKE UP --")

for i in [m_ben_simben, w_ben_simben]:
	if i[0] < 0.01:
		sig = "1%"
	elif i[0] < 0.05:
		sig = "5%"
	elif i[0] < 0.10:
		sig = "10%"
	else:
		sig = "N/A"
	print(f"Coefficient on SimBenefit is signficant at {sig} for {i[1]}")

print("Average Take Up Rate Elasticity: ", round(epsilon_bar, 5))
print("Average Individual Change in Take Up: ", round(avg, 5))
print("Overall Change in Take Up: ", round(pct_change_total, 5))
print("Total Take Up Under Baseline: ", round(baseline_takeup, 5))
print("Total Take Up Under New: ", round(baseline_takeup + totalchange, 5))

print("   ")



print("-- FISCAL EFFECTS --")

print("Average Baseline Benefit Value: ", round(df['SIMBENEFIT'].mean(), 5))
print("Average New Benefit Value: ", round(df['SIMBENEFIT_NEW'].mean(), 5))
print("Cost of Baseline Program Benefits: ", round(sum(df['SIMBENEFIT']), 5))
print("Cost of New Program Benefits: ", round(sum(df['SIMBENEFIT_NEW']), 5))

print("   ")
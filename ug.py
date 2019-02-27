import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
from sklearn.ensemble import RandomForestClassifier

yes_no_map = {'yes':1, 'no': 0}
job_map = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
			'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9,
			'unemployed': 10}
marital_map = {'divorced': 0, 'married': 1, 'single': 2}
education_map = {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 'illiterate': 4,
				'professional.course': 5, 'university.degree': 6}
contact_map = {'cellular': 0, 'telephone': 1}
month_map = {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6,
			'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11}
day_map = {'sun': 0, 'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6}
poutcome_map = {'failure': 0, 'success': 1}

def cleanData ():
	df['job'] = df['job'].map(job_map)
	df['marital'] = df['marital'].map(marital_map)
	df['education'] = df['education'].map(education_map)
	df['contact'] = df['contact'].map(contact_map)
	df['month'] = df['month'].map(month_map)
	df['day_of_week'] = df['day_of_week'].map(day_map)
	df['poutcome'] = df['poutcome'].map(poutcome_map)
	df['default'] = df['default'].map(yes_no_map)
	df['housing'] = df['housing'].map(yes_no_map)
	df['loan'] = df['loan'].map(yes_no_map)
	df['y'] = df['y'].map(yes_no_map)
	col_names = list(df)
	for column in col_names:
		df[column] = df[column].apply(lambda x: np.random.choice(df[column].dropna().values) if np.isnan(x) else x)

def myBarPlot (col_name):
	c_dict = {'job': 11, 'marital': 3, 'education': 7, 'contact': 2, 'month': 12,
	 'day_of_week': 7, 'poutcome': 2, 'default': 2, 'housing': 2, 'loan': 2}
	c = c_dict[col_name]
	x = np.zeros(c)
	y_good = np.zeros(c); y_bad = np.zeros(c);
	for m in range (c):
		good = df.loc[(df[col_name] == m) & (df['y'] == 1)]
		bad = df.loc[(df[col_name] == m) & (df['y'] == 0)]
		x[m] = m; y_good[m] = len(good); y_bad[m] = len(bad)
	y_rat = np.divide(y_good,(y_good+y_bad+1e-5));
	plt.subplot(2,1,1); graph = plt.bar(x,y_good,color='green'); plt.title('Yes',loc='right',color='green'); plt.xlabel(col_name); plt.ylabel('count');
	plt.subplot(2,1,2); graph = plt.bar(x,y_bad,color='red'); plt.title('No',loc='right', color='red'); plt.xlabel(col_name); plt.ylabel('count');
	# plt.subplot(3,1,3); graph = plt.bar(x,y_rat); plt.title('Yes Ratio')
	plt.show()

def myHist (col_name, b):
	global df
	if (col_name == 'pdays'):
		df = df.loc[~(df['pdays'] == 999)]
	good = df[col_name].loc[(df['y'] == 1)]
	bad = df[col_name].loc[(df['y'] == 0)]
	plt.subplot(2,1,1)
	good_hist = good.hist(bins=b, grid=False, color='green'); plt.title('Yes',loc='right',color='green'); plt.xlabel(col_name); plt.ylabel('count')
	plt.subplot(2,1,2); 
	bad_hist = bad.hist(bins=b, grid=False, color='red'); plt.title('No',loc='right', color='red'); plt.xlabel(col_name); plt.ylabel('count') 
	plt.show();
	
def visualize ():
	# myBarPlot('education') # illiterate (4) has very less good... but no use, only one illiterate
	# myBarPlot('job') # blue collar ratio very less, student ratio quite high
	# myBarPlot('marital') # useless
	# myBarPlot('month') # nothing interesting
	# myBarPlot('default') # useless, defaulters are nonexistent
	# myBarPlot('housing') # useless
	# myBarPlot('loan') # useless
	# myBarPlot('contact') # telephone users
	# myBarPlot('poutcome') # no idea

	# myHist('duration', 10) # Possibly useless
	# myHist ('pdays', 10) # Can use this, pdays > 13, bad is much more

	# ax = sns.lmplot(x = 'age', y = 'y', data = df, truncate = True)
	ax = sns.lmplot(x='campaign', y='y', order=1, data=df, truncate=True, fit_reg=False, scatter_kws={'color': 'violet'}) # important
	# ax = sns.lmplot(x='nr.employed', y='y', order=1, data=df, truncate=True)
	# ax = sns.lmplot(x='duration', y='y', order=1, data=df, truncate=True, fit_reg=False)
	# ax = sns.lmplot(x='euribor3m', y='y', order=1, data=df, truncate=True) # important, roughly 3-4 range can be omitted
	# ax = sns.lmplot(x='emp.var.rate', y='y', order=1, data=df, truncate=True)
	# ax = sns.lmplot(x='pdays', y='y', order=1, data=df.loc[~(df['pdays'] == 999)], truncate=True)
	# ax = sns.lmplot(x='previous', y='y', order=1, data=df.loc[~(df['pdays'] == 999)], truncate=True)
	plt.show()

def campaignStuff ():
	conv_rat = sum(df['y'])/sum(df['campaign'])
	print ('Conversion ratio :', conv_rat)
	for k in range (1,30):
		goo = sum(df.loc[df['campaign']==k]['y']) / float(df.loc[df['campaign'] >= k].shape[0])
		total_calls = sum(df['campaign']) # total calls
		a = sum(df.loc[(df['campaign'] > k)]['campaign']) # Called more than 6 times
		b = k*df[df['campaign'] > k].shape[0] # Called more than 6 times, 6*number of instances
		extra_calls = a-b
		reduction_percent = extra_calls/total_calls*100
		total_sales = df[df['y']==1].shape[0] # Obvious
		less_costly_sales = df[(df['campaign'] <= k) & (df['y']==1)].shape[0] # Favourable sales
		sales_percent=100*less_costly_sales/total_sales
		print (k, reduction_percent, sales_percent, goo)

def euribor3mStuff ():
	conv_rat = sum(df['y'])/sum(df['campaign'])
	df1 = df.loc[(df['euribor3m'] < 3) | (df['euribor3m'] > 4)]
	print (df1.shape, df.shape)
	conv_rat1 = sum(df1['y'])/sum(df1['campaign'])
	print (conv_rat1, conv_rat)

def jobStuff ():
	conv_rat = sum(df['y'])/sum(df['campaign'])
	df1 = df.loc[ (df['job'] != 2) & (df['job'] != 993)] # 2,3,6
	total_calls = sum(df['campaign'])
	new_calls = sum(df1['campaign'])
	extra_calls = total_calls - new_calls
	reduction_percent = extra_calls/total_calls*100
	total_sales = df[df['y']==1].shape[0] # Obvious
	less_costly_sales = df1[df1['y']==1].shape[0] # Favourable sales in new
	sales_percent=100*less_costly_sales/total_sales
	print (reduction_percent, sales_percent)

def RF ():
	M = df.values
	X = M[:,:-1]; Y = M[:,-1]
	clf = RandomForestClassifier(n_estimators=100, verbose=1, class_weight='balanced')
	clf.fit(X,Y)
	col_names = list(df)
	print ('Feature importances:')
	for i in range (X.shape[1]):
		print (col_names[i], clf.feature_importances_[i])


# df = pd.read_csv('./bank-additional-full.csv', sep=';')
# cleanData()
# df.to_csv('bank-edited.csv', index=None, header=True)

df = pd.read_csv('bank-edited.csv')

# visualize()

campaignStuff()
# euribor3mStuff()
# pdaysStuff()
# jobStuff()

# RF()
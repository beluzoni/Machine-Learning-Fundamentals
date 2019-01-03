import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Create your df here:Anthony Graves 1-3-2019

We want to see:

at least two graphs containing exploration of the dataset
a statement of your question (or questions!) and how you arrived there
the explanation of at least two new columns you created and how you did it
the comparison between two classification approaches, including a qualitative discussion of simplicity, time to run the model, and accuracy, precision, and/or recall
the comparison between two regression approaches, including a qualitative discussion of * simplicity, time to run the model, and accuracy, precision, and/or recall
an overall conclusion, with a preliminary answer to your initial question(s), next steps, and what other data you would like to have in order to better answer your question(s)

#########################################
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
df = pd.read_csv("profiles.csv")
cols = df.shape[1]
rows = df.shape[0]
pd.set_option('display.max_columns', cols*2)

########################################
df.tail(1)

#
import json
with open ("us_states.json", 'r') as f:
    us_state_abbrev = json.load(f)
#

Adding new columns - city, state, in the US, in CA, longtitude, and latitude
I wanted to add geolocation data for geomapping, but along the way found out that almost all of the data was in California. Because of this, I just got geolocation data for California cities.

It wasn't possible using the api I found to get geolocation data for all cities, but I was able to for almost all of them.

#
df['city'] = df.location.map(lambda l : l.split(", ")[0])
df['state'] = df.location.map(lambda l : l.split(", ")[1])
df['stateAbbr'] = df.state.map(lambda s : us_state_abbrev[s] if s in us_state_abbrev else s)
df['inUS'] = df.state.map(lambda s : s in us_state_abbrev)
df['inCA'] = df.state.map(lambda s : s == 'california')

inUS = df.inUS.value_counts()
inCA = df.inCA.value_counts()
print("{}% in US".format(inUS[1]/rows*100))
print("{}% in CA".format(inCA[1]/rows*100))

Because 99.978% of the data are in the US and 99.848% of the data are in California, all non-California observations are going to be ignored for any geo-location summary information.

#########################################

import requests
def getLatLong(location):
    print(".", end="")
    response = requests.get("http://open.mapquestapi.com/geocoding/v1/address?key=pxVoRtOLk6bTqJAGmcCdLY9ZrLGA707h&location={}".format(location))  
    try:
        response = response.json()
        state = location.split(", ")[1]
        latLong = None
        for data in response['results'][0]['locations']:
            if state == data['adminArea3']:
                latLong = data['latLng']
        if not latLong: # could not find the actual state's data
            raise
        lat = latLong['lat']
        long = latLong['lng']
        return lat, long
    except:
        print(" {} missing ".format(location), end="")
        return None, None
    
################################
    
    locations = df.location.value_counts().index
location_counts = list(df.location.value_counts())
location_dict = {}

for location, count in zip(locations,location_counts):
    city = location.split(", ")[0]
    state = location.split(", ")[1]
    if not state == 'california':
        continue
    lat, long = getLatLong("{}, CA, US".format(city))
    if not lat:
        continue
    location_dict[city] = {"count": count, "lat": lat, "long": long}
    
#########################
    
    df['lat'] = df.city.map(lambda c : location_dict[c]['lat'] if c in location_dict else None)
df['long'] = df.city.map(lambda c : location_dict[c]['long'] if c in location_dict else None)

########################

df.to_csv("profilesWithGeoData.csv")
import json
with open('location_dict.json', 'w') as f:
    json.dump(location_dict, f)
    
    
    
### START FROM HERE: Start from here in the future (no need to redo getting geolocation data) ######

import pandas as pd
import numpy as np
import sys, json
from matplotlib import pyplot as plt

with open('location_dict.json', 'r') as f:
    location_dict = json.load(f)

df = pd.read_csv("profilesWithGeoData.csv", index_col=0)
cols = df.shape[1]
rows = df.shape[0]
pd.set_option('display.max_columns', cols*2)
df.head(1)

###### First 'graph' - listing density by geolocation ##############
# The map below shows the density of listings by geolocation. The log of the number of listings per location was taken, as otherwise #everything but San Francisco would be white.
#(the scale for the map markers is white to red, white being low density of listings and red being high)



!{sys.executable} -m pip install -q folium
import folium
import math


MAX_COUNT = math.log(list(df.location.value_counts())[0])
MIN_LISTINGS = 10
color_scaler = (MAX_COUNT)/255
def percToRed(perc):
    return '#%02x%02x%02x' % (255, 255-int(perc*255), 255-int(perc*255))

# map centered on US
dating_map = folium.Map(location=[36.7783, -119.4179], zoom_start=6)

for location, data in location_dict.items():
    count = data['count']
    if count < MIN_LISTINGS:
        continue # ignore locations with less than MIN_LISTINGS listings
    percent = math.log(count)/MAX_COUNT
    color = percToRed(percent)
    folium.CircleMarker(
        [data['lat'], data['long']],
        radius=5,
        popup=folium.Popup("{} ({} listings)".format(location, count), parse_html=True),
        fill=True,
        color=color,
        fill_color=color,
        fill_opacity=0.6
        ).add_to(dating_map)
    
### Body: Fitness Map
body_map = {"thin":0, "skinny":1, "athletic":2,"fit":3, "jacked": 4, "average":5, "a little extra":6, 
            "curvy":7, "full figured":8, "used up": 9,  "overweight": 10, "rather not say": 11 }
df.body_type.value_counts()

### Body Fitness Map bar graph

ax = df.body_type.value_counts().plot(kind='bar')
ax.set_xlabel('Weight category')
ax.set_ylabel('Number of respondents')

##### Drink Map

drinks_map = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df.drinks.value_counts()

### Drinks Bar Graph
ax = df.drinks.value_counts().plot(kind='bar')
ax.set_xlabel('Drinks category')
ax.set_ylabel('Number of respondents')

### Drugs
drugs_map = {"never": 0, "sometimes": 1, "often": 2}
df.drugs.value_counts()

## Drugs Map

ax = df.drugs.value_counts().plot(kind='bar')
ax.set_xlabel('Drugs category')
ax.set_ylabel('Number of respondents')

## Smokes
smokes_map = {"no": 0, "when drinking": 1, "sometimes": 2, "trying to quit": 3, "yes": 4 }
df.smokes.value_counts()

## Male/Female Ratio
sex_map = {"f": 0, "m": 1}
df.sex.value_counts()


#Education Map
education_map = {'dropped out of high school': 0, 'dropped out of space camp': 1, 'working on high school': 2, 'high school': 3, 'graduated from high school': 4, 'working on space camp': 5, 'space camp': 6, 'graduated from space camp': 7, 'dropped out of two-year college': 8, 'dropped out of college/university': 9, 'working on two-year college': 10, 'two-year college': 11, 'graduated from two-year college': 12, 'working on college/university': 13, 'college/university': 14, 'graduated from college/university': 15, 'dropped out of masters program': 16, 'dropped out of law school': 17, 'dropped out of med school': 18, 'dropped out of ph.d program': 19, 'working on masters program': 20, 'masters program': 21, 'graduated from masters program': 22, 'working on law school': 23, 'law school': 24, 'graduated from law school': 25, 'working on med school': 26, 'med school': 27, 'graduated from med school': 28, 'working on ph.d program': 29, 'ph.d program': 30, 'graduated from ph.d program': 31}


##
ax = df.education.value_counts().plot(kind='bar')
ax.set_xlabel('HIghest Educational Achievement Category')
ax.set_ylabel('Number of respondents')

##
df.income.value_counts()


#
ax = df.income.value_counts().plot(kind='bar')
ax.set_xlabel('Income Band')
ax.set_ylabel('Number of respondents')

# Difference in reported income Lambda
df['income_reported'] = df.income.map(lambda i: 0 if i < 0 else 1)
df['income'] = df.income.map(lambda i: 0 if i < 0 else i)

#Correlation matrix between quantifiable features of the dataset
from matplotlib.pyplot import figure
import seaborn as sns

correlation_data = df[['age', 'height', 'income', 'income_reported', 'drinks_scale', 'drugs_scale', 'smokes_scale', 'sex_scale', 'education_scale']]
correlation_data.columns = ['Age', 'Height', 'Income', 'Income Reported', 'Drinking Propensity', 'Drugs Propensity', 'Smoking Propensity', 'Gender', 'Education Level']

f, ax = plt.subplots(figsize=(10, 8))
corr = correlation_data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot=True)


##Pairwise plots - click on left bar to the left of plots to expand all
import seaborn
seaborn.pairplot(correlation_data.dropna(), kind="reg")


##Dealing with NaN values
#For drinking, drugs, smoking, and education, there were a good number of missing values. It is likely that non-reporting of these values has some bias inherent; my guess is people who drink or use or smoke would be slightly more likely to not respond than those who do not.

#Because of this and the inability to measure the degree of the bias (this is a self-reported survey, there is no way to find out the degree of the bias), it is safer to drop any observations with NaN values than to estimate their values.
final_df = correlation_data.dropna()
final_df = final_df.reset_index()
final_df = final_df.drop('index', axis=1)
print("Before dropping: {} observations.\nAfter dropping: {} observations".format(
    correlation_data.shape[0],final_df.shape[0]))

#
print(final_df['Income Reported'].value_counts())
final_df.tail()



###   Research question and how I arrived there
#The fact that most people do not report their income is unsurprising given its private nature and our culture (in many cultures people are much more open about their income), but it is also quite intriguing to consider whether other quantifiable factors are related to the reporting of income or not, especially after viewing the many correlations with reported income that exist in the data.

#So, my research question came to be:

#Can reported income be accurately classified and what are the factors that contribute to whether it was reported or not?

#Unfortunately, the dataset is entirely self-reported, so our confidence in the results in any question posed will be undermined by self-reporting bias, but there is nothing to be done about that.

#Since we are required to not only do classification but regression, I have decided to use KNN and SVM for classification, and logistic regression and multiple linear regression for regression.

#Since the data does not need to be normalized for logistic or multiple linear regression and interpretation will be clearer without normalization, we will start there.
final_df = correlation_data.dropna()
final_df = final_df.reset_index()
final_df = final_df.drop('index', axis=1)
print("Before dropping: {} observations.\nAfter dropping: {} observations".format(
    correlation_data.shape[0],final_df.shape[0]))


###
!{sys.executable} -m pip install -q scikit-plot

#####
def showConfusionMatrixAndMore(model, model_type, X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model.fit(X_train, y_train)
    y_predictions = model.predict(X_test)
    y_predictions = [0 if pred < 0.5 else 1 for pred in y_predictions]
    confmat = confusion_matrix(y_true=y_test, y_pred=y_predictions)
    
    fig, ax = plt.subplots(figsize=(2.5,2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix for {}'.format(model_type), y=1.2)
    plt.show()
    precision = precision_score(y_true=y_test, y_pred=y_predictions)
    recall = recall_score(y_true=y_test, y_pred=y_predictions)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_predictions)
    print("""
    Precision was: {}
    Accuracy was: {}
    Recall was: {}
    """.format(precision,accuracy,recall))
    
    
    ###
    from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

def showROC(classifier, model_type, X, y):
    plt.figure(figsize=(7,7))
    cv = StratifiedKFold(n_splits=6)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        fit = classifier.fit(X.loc[train], y[train])
        if 'predict_proba' in dir(fit): # model has a predict probabilities function            
            probas_ = [prob[1] for prob in fit.predict_proba(X.loc[test])]
        else: # model only has a predict function
            probas_ = [0 if prob < 0.5 else 1 for prob in fit.predict(X.loc[test])]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for {}'.format(model_type))
    plt.legend(loc="lower right")
    plt.show()
    
    ## Logistic Regression
    logit = LogisticRegression()
showConfusionMatrixAndMore(logit, 'Logistic Regression', X, y)

##
showROC(logit,'Logistic Regression', X,y)


## Multiple Linear Regression

from sklearn.linear_model import LinearRegression

linear = LinearRegression()
showConfusionMatrixAndMore(linear, 'Linear Regression', X, y)

###
showROC(linear,'Linear Regression', X,y)

##
import statsmodels.discrete.discrete_model as sm

linear_reg = sm.OLS(y, X)
linear_reg.fit().summary()

##K-Nearest Neighbors
normalized_X=(X-X.mean())/X.std()

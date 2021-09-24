#outlier detection
outliers = []
def detect_outliers_iqr(data):
    data = sorted(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    # print(q1, q3)
    IQR = q3-q1
    lwr_bound = q1-(1.5*IQR)
    upr_bound = q3+(1.5*IQR)
    # print(lwr_bound, upr_bound)
    for i in data: 
        if (i<lwr_bound or i>upr_bound):
            outliers.append(i)
    return len(outliers)# Driver code
sample_outliers = detect_outliers_iqr(new_data.f1)
print("Outliers from IQR method: ", sample_outliers)

for sample in new_data.columns.values:
    # Computing 10th, 90th percentiles and replacing the outliers
    tenth_percentile = np.percentile(new_data[sample], 10)
    ninetieth_percentile = np.percentile(new_data[sample], 90)
    new_data[sample] = np.where(new_data[sample]<tenth_percentile, tenth_percentile, new_data[sample])
    new_data[sample] = np.where(new_data[sample]>ninetieth_percentile, ninetieth_percentile, new_data[sample])
    
    
for sample in new_test.columns.values:
    # Computing 10th, 90th percentiles and replacing the outliers
    tenth_percentile = np.percentile(new_test[sample], 10)
    ninetieth_percentile = np.percentile(new_test[sample], 90)
    new_test[sample] = np.where(new_test[sample]<tenth_percentile, tenth_percentile, new_test[sample])
    new_test[sample] = np.where(new_test[sample]>ninetieth_percentile, ninetieth_percentile, new_test[sample])    
    
    
#kfold
from sklearn import model_selection
# we create a new column called kfold and fill it with -1
df["kfold"] = -1
# the next step is to randomize the rows of the data
df = df.sample(frac=1).reset_index(drop=True)
# initiate the kfold class from model_selection module
kf = model_selection.KFold(n_splits=5)
# fill the new kfold column
for fold, (trn_, val_) in enumerate(kf.split(X=df)):
    df.loc[val_, 'kfold'] = fold
    
    
#run k fold
def fold_run(fold):
    # training data is where kfold is not equal to provided fold
    # also, note that we reset the index
    df_train = df[df.kfold != fold].reset_index(drop=True)
    # validation data is where kfold is equal to provided fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    # drop the label column from dataframe and convert it to
    # a numpy array by using .values.
    # target is label column in the dataframe
    x_train = df_train.drop("claim", axis=1).values
    y_train = df_train.claim.values
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    # similarly, for validation, we have
    x_valid = df_valid.drop("claim", axis=1).values
    x_valid = scaler.transform(x_valid)
    y_valid = df_valid.claim.values
    # initialize simple decision tree classifier from sklearn
    clf = xgb.XGBClassifier()
    # fir the model on training data
    clf.fit(x_train, y_train)
    # create predictions for validation samples
    preds = clf.predict(x_valid)
    # calculate & print accuracy
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    
    
from sklearn.impute import SimpleImputer
# make copy to avoid changing original data (when Imputing)
new_data = data.copy()
new_test = data_test.copy()
new_data = new_data.drop('claim', axis = 1)
# Imputation
my_imputer = SimpleImputer()
new_data = my_imputer.fit_transform(new_data)
new_test = my_imputer.transform(new_test)

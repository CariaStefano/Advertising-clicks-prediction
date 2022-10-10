import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

if __name__=='__main__':
    df= pd.read_csv('datasets/advertising.csv')
    df.columns= ['Daily_Time_Spent_on_Site', 'Age', 'Area_Income',
    'Daily_Internet_Usage', 'Ad_Topic_Line', 'City', 'Male', 'Country',
    'Timestamp', 'Clicked_on_Ad']

#tipologia features
    print(df.info())

    #oppure

    print("Tipologia delle features: \n")
    print("Daily_Time_Spent_on_Site: ", df.Daily_Time_Spent_on_Site.dtype)
    print("Age: ",df.Age.dtype)
    print("Area_Income: ", df.Area_Income.dtype)
    print("Daily_Internet_Usage: ", df.Daily_Internet_Usage.dtype)
    print("Ad_Topic_Line: ",df.Ad_Topic_Line.dtype)
    print("City: ", df.City.dtype)
    print("Male: ", df.Male.dtype)
    print("Country: ",df.Country.dtype)
    print("Timestamp: ", df.Timestamp.dtype)
    print("Clicked_on_Ad: ", df.Clicked_on_Ad.dtype)

#b. Rappresentazione numerica (media, mediana, standard deviationquartili,moda)
    print("\n\n")
    for key, value in df.iteritems():
        if df[key].dtype== 'int64' or df[key].dtype== 'float64' :
            print(df[key].name)
            print("median \t\t", df[key].median())
            print("mode \t\t", df[key].mode().unique())
            print(df[key].describe(), "\n")

# c. Rappresentazione grafica (se possibile)

    df['Male']= df['Male'].map({1: 'Male', 0:'Female'})
    df['Clicked_on_Ad']= df['Clicked_on_Ad'].map({1: 'click', 0:'not click'})
    contincency_table= pd.crosstab(df.Clicked_on_Ad, df.Male)
    print(contincency_table)

    contincency_table.plot.pie(y='Female')
    plt.show()
    contincency_table.plot.pie(y='Male') #non c'è una grossa differenza tra uomo e donna per i click sulla pubblicità
    plt.show()

#riutiliziamo la funzione map per ripristinare i valori di partenza perchè sarà utile nella fase di previsione.
    df['Male']= df['Male'].map({'Male': 1, 'Female':0})
    df['Clicked_on_Ad']= df['Clicked_on_Ad'].map({'click': 1, 'not click':0})

####
    sns.displot(df['Age'], kde='True') #Dal grafico possiamo assumere che la variabile 'Age' ha una distribuzione normale
    plt.show()

    sns.displot(x='Age' , y='Daily_Time_Spent_on_Site' ,data=df, kind='kde', rug=True, )
    plt.show() #il grafico mostra che gli utenti più giovani (25-40 anni) sono quelli che  spendono più tempo sul sito

    f, ax = plt.subplots(figsize=(8, 8))
    cmap = sns.cubehelix_palette(as_cmap=True, start=0, dark=0, light=3, reverse=True)
    sns.kdeplot(df["Daily_Time_Spent_on_Site"], y=df["Daily_Internet_Usage"],
                cmap=cmap, n_levels=100, shade=True);
    plt.show()

    sns.displot(x=df['Daily_Time_Spent_on_Site'], y=df["Daily_Internet_Usage"])




    scatter_matrix(df[["Daily_Time_Spent_on_Site", 'Age', 'Area_Income', 'Daily_Internet_Usage']],
                   alpha=0.3, figsize=(10, 10));
    plt.show()

#Preprocessing
    object_variables = ['Ad_Topic_Line', 'City', 'Country']
    print(df[object_variables].describe(include=['O'])) #come già osservato in precedenza, le variabili City e Ad_Topic_Line
                                                        # contengono troppe osservazioni uniche (rispettivamente 969 e 1000).
                                                        # Poichè in genere è molto dificile fare previsioni senza che esista
                                                        # un data patern queste variabile saranno escluse. La variabile Country ha
                                                        # un unico elemento ripetuto 9 volte (France). Esploriamo meglio la variabile.


    print(df["Country"].value_counts()[:15]) # nel dataset sono presenti 237 countries ma non è presente un country dominante
                                            # Tanti elementi unici non consentirà a un modello di machine learning di stabilire
                                            # relazioni. per questo motivo anche Country verrà esclusa.

    df = df.drop(['Ad_Topic_Line', 'City', 'Country'], axis=1) #escludiamo le variabili


    df['Timestamp'] = pd.to_datetime(df['Timestamp'])   # si è proceduto a dividere la variabile "timestamp" in:
                                                        # "Month", "Day of manth", "Day of week", "Hour". In questo modo
                                                        # si possono agevolmente elaborare le informazioni sul giorno del
                                                        #mese, della settimana e l'ora in cui la pubblicità è stata cliccata


    df['Month'] = df['Timestamp'].dt.month
    df['Day of month'] = df['Timestamp'].dt.day
    df['Day of week'] = df['Timestamp'].dt.dayofweek
    df['Hour'] = df['Timestamp'].dt.hour
    df = df.drop(['Timestamp'], axis=1)

    print(df.head())

#dividiamo il dataset in training e test set. Tutte le variabili, a eccezione di "Clicked_on_Ad"(variabile da predire-->y) saranno le variabili input (x)
    X = df[['Daily_Time_Spent_on_Site', 'Age', 'Area_Income', 'Daily_Internet_Usage',
              'Male', 'Month', 'Day of month', 'Day of week']]
    y = df['Clicked_on_Ad']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

#il primo modello importato sarà un Logistic Regression model.
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix

    model1 = LogisticRegression()
    model1.fit(X_train_std, y_train)
    predictions_LR = model1.predict(X_test_std)

    print('\nLogistic regression accuracy %.5f:'% accuracy_score(predictions_LR, y_test))
    print('\nConfusion Matrix:')
    print(confusion_matrix(predictions_LR, y_test))
    print('Misclassified samples: %d' % (y_test != predictions_LR).sum())


#DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier

    model2 = DecisionTreeClassifier()
    model2.fit(X_train_std, y_train)
    predictions_DT = model2.predict(X_test_std)

    print('\nDecision Tree accuracy: %.5f'% accuracy_score(predictions_DT, y_test))
    print('\nConfusion Matrix:')
    print(confusion_matrix(predictions_DT, y_test))
    print('Misclassified samples: %d' % (y_test != predictions_DT).sum())


#Perceptron
    from sklearn.linear_model import Perceptron

    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    from sklearn.linear_model import Perceptron

    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    print(y_test.shape)
    y_pred = ppn.predict(X_test_std)

    print('Perceptron Accuracy: %.5f' % accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_pred, y_test))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    clf = svm.SVC(kernel='linear', C=1, random_state=0).fit(X_train_std, y_train)
    print(clf.score(X_test_std, y_test))
    scores = cross_val_score(clf, X, y, cv=5)
    print("cross validation =5 --> accuracy Perceptron \n",scores) # testiamo il modello con una CV=5.


import pandas as pd


df = pd.read_csv('spam.csv')
df.head()

df.groupby('Category').describe()

df['spam'] = df['Category'].apply(lambda x:1 if x == 'spam' else 0)
df.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.25)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:3]


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)

emails = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
           "Nah I don't think he goes to usf, he lives around here though"
          ]

emails_count = v.transform(emails)
model.predict(emails_count)          

X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)

from sklearn.pipeline import Pipeline
clf = Pipeline([
       ('vectorizer', CountVectorizer()),
       ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

clf.score(X_test, y_test)

clf.predict(emails)
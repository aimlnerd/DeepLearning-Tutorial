from dl_regression import DLRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()

type(boston)
features = boston['data']
feature_names = boston['feature_names']
target = boston['target']
target = target.reshape(target.shape[0],1)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=4585)

dlreg = DLRegression(hidden_layers=[4,2],
                     batch_size=None,
                     dropout=0.1,
                     l2_regularizer=1e-05,
                     learning_rate=0.01,
                     beta1=0.9,
                     beta2=0.99,
                     epsilon=1e-08,
                     epochs=100,
                     save_path='./save/model',
                     logs_path='./log',
                     print_epoch=True)

dlreg.fit(X=X_train, y=y_train)
y_test_pred = dlreg.predict(X=X_test)

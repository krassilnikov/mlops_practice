# задание №1
python-скрипт (data_creation.py) - создает несколько наборов данных. В некоторые данные вклены аномалии и шумы. Часть наборов данных сохраняется в папке «train», другая часть — в папке «test».
python-скрипт (model_preprocessing.py) - выполняет предобработку данных с помощью sklearn.preprocessing.StandardScaler.
python-скрипт (model_preparation.py) - создает и обучает модель машинного обучения на построенных данных из папки «train».
python-скрипт (model_testing.py) - проверяет модель машинного обучения на построенных данных из папки «test».
bash-скрипт (pipeline.sh) - последовательно запускает все python-скрипты.

#PS
Убедитесь, что у вас установлены необходимые библиотеки:
pip install numpy pandas scikit-learn joblib
Сделайте bash-скрипт исполняемым:
chmod +x pipeline.sh

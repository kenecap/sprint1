Проект автодополнение текста
sprint1/
├── data/                            # Датасеты
│   ├── tweets.txt                   # сырой датасет
│   ├── train.csv                    # тренировочная выборка
│   ├── val.csv                      # валидационная выборка
│   └── test.csv                     # тестовая выборка
│
├── src/                             # Весь код проекта
│   ├── data_utils.py                # Обработка датасета
|   ├── next_token_dataset.py        # код с torch Dataset'ом 
│   ├── lstm_model.py                # код lstm модели
|   ├── eval_lstm.py                 # замер метрик lstm модели
|   ├── lstm_train.py                # код обучения модели
|   ├── eval_transformer_pipeline.py # код с запуском и замером качества трансформера
│
├── models/                          # веса обученных моделей
|
├── solution.ipynb                   # ноутбук с решением
└── requirements_sprint_2_projec     # зависимости проекта
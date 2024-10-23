from transformers import pipeline
import pandas as pd

def format_inputs(data):
    if isinstance(data, str):
        return data

    def format_dict(data):
        return f"{data['channel_name']} [{data['reel_name']}] {data['description']}"
    if isinstance(data, dict):
        return format_dict(data)
    if isinstance(data, list):
        if not data or isinstance(data[0], str):
            return data
        if isinstance(data[0], dict):
            return [format_dict(x) for x in data]

    def format_pandas(data):
        return f'{data.yt_channel_name} {data.text}'
    if isinstance(data, pd.Series):
        return format_pandas(data)
    if isinstance(data, pd.DataFrame):
        return data.apply(format_pandas, axis=1).tolist()

class ReelsClassifier:
    def __init__(self, model='daniilxcode/reels-clf', **kwargs):
        self.clf = pipeline('text-classification', model, **kwargs)

    def predict(self, inputs, **kwargs):
        """
        Предсказать к какому проекту относятся входные данные.

        Формат входных данных:
        1. `dict` состоящий из 'channel_name', 'reel_name' и 'description'
        2. `list` из `dict` такого же формата
        3. `DataFrame` с колонками 'yt_channel_name' и 'text'
        4. Отдельная строка датафрейма (`Series` имеющий 'yt_channel_name' и 'text')

        Возвращает `list` из `dict` с ключами 'label' и 'score'.
        Количество элементов в возвращаемом списке равно количеству входных данных.
        'label' - Проект к которому модель относит данные
        'score' - Уверенность модели
        """
        inputs = format_inputs(inputs)
        return self.clf(inputs, truncation=True, **kwargs)

    def predict_proba(self, inputs, **kwargs):
        inputs = format_inputs(inputs)
        return self.clf(inputs, truncation=True, top_k=-1, **kwargs)

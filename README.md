# Классификация анимационных проектов

Пример:
```py
from reels_clf import ReelsClassifier

clf = ReelsClassifier()

inputs = {
    'channel_name': 'Название канала',
    'reel_name': 'Название ролика',
    'description': 'Описание ролика'
}
predictions = clf.predict(inputs)

print(predictions[0]['label'])
```

Запуск демо: `streamlit run demo.py`

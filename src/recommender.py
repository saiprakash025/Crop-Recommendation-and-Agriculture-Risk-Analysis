import numpy as np


def recommend(model, label_encoder, frame, top_n=3):
    proba = model.predict_proba(frame)
    results = []
    for row in proba:
        order = np.argsort(row)[::-1][:top_n]
        names = label_encoder.inverse_transform(model.classes_[order])
        results.append([(str(n), round(float(row[i]) * 100, 1)) for n, i in zip(names, order)])
    return results

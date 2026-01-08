import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# =========================
# 1. Reproducibility
# =========================
np.random.seed(42)
tf.random.set_seed(42)

# =========================
# 2. Configuration
# =========================
VOCAB_SIZE = 1000
MAX_LEN = 6
EMBEDDING_DIM = 16
GRU_UNITS = 32
EPOCHS = 30
TEST_SIZE = 0.2

# =========================
# 3. Dataset
# =========================
texts = [
    "I love this movie",
    "This product is amazing",
    "I am very happy",
    "I hate this",
    "This is terrible",
    "I am very sad"
]

labels = np.array([1, 1, 1, 0, 0, 0])  # 1 = Positive, 0 = Negative

# =========================
# 4. Text Preprocessing
# =========================
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)


X_train, X_val, y_train, y_val = train_test_split(
    padded_sequences,
    labels,
    test_size=TEST_SIZE,
    random_state=42
)


model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
    GRU(GRU_UNITS),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)


model.save("gru_sentiment_model.h5")


def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    probability = model.predict(padded)[0][0]

    return "Positive 😊" if probability > 0.5 else "Negative 😞"


test_sentence = "I really love this product"
print("Sentence:", test_sentence)
print("Prediction:", predict_sentiment(test_sentence))

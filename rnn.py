import random
import math

vocabulary = ["dr", "noha", "is", "beautiful"]
word_to_index = {word: i for i, word in enumerate(vocabulary)}
index_to_word = {i: word for word, i in word_to_index.items()}

def one_hot_encode(word, vocab_size):
    vec = [0] * vocab_size
    if word in word_to_index:
        vec[word_to_index[word]] = 1
    return vec

input_words = ["dr", "noha", "is"]
target_word = "beautiful"

input_size = len(vocabulary)
hidden_size = 5
output_size = len(vocabulary)
learning_rate = 0.01
epochs = 100  

Wxh = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(hidden_size)]
Whh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
Why = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(output_size)]
bh = [0.0] * hidden_size
by = [0.0] * output_size

def tanh(x):
    return [(2 / (1 + (2.71828 ** (-2 * xi)))) - 1 for xi in x]

def tanh_derivative(x):
    return [1 - xi ** 2 for xi in x]

def softmax(x):
    exp_x = [2.71828 ** i for i in x]
    sum_exp = sum(exp_x)
    return [i / sum_exp for i in exp_x]

def dot(matrix, vector):
    return [sum(m * v for m, v in zip(row, vector)) for row in matrix]

def add(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

inputs = [one_hot_encode(w, input_size) for w in input_words]
target = one_hot_encode(target_word, output_size)

for epoch in range(epochs):
    hidden_states = []
    hidden = [0.0] * hidden_size
    hidden_states.append(hidden)

    for x in inputs:
        hidden_input = add(dot(Wxh, x), dot(Whh, hidden))
        hidden = tanh(add(hidden_input, bh))
        hidden_states.append(hidden)

    output_raw = add(dot(Why, hidden), by)
    output_probs = softmax(output_raw)

    loss = -sum(t * math.log(p + 1e-8) for t, p in zip(target, output_probs))
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {round(loss, 4)}")

    dWhy = [[0.0] * hidden_size for _ in range(output_size)]
    dby = [0.0] * output_size
    dWxh = [[0.0] * input_size for _ in range(hidden_size)]
    dWhh = [[0.0] * hidden_size for _ in range(hidden_size)]
    dbh = [0.0] * hidden_size

    output_error = [output_probs[i] - target[i] for i in range(output_size)]
    for o in range(output_size):
        for h in range(hidden_size):
            dWhy[o][h] += output_error[o] * hidden_states[-1][h]
        dby[o] += output_error[o]

    dh_next = [0.0] * hidden_size
    for t in reversed(range(len(inputs))):
        dh = [sum(output_error[o] * Why[o][h] for o in range(output_size)) + dh_next[h]
              for h in range(hidden_size)]
        dh_raw = [dh[i] * tanh_derivative(hidden_states[t+1])[i] for i in range(hidden_size)]

        for h in range(hidden_size):
            for i in range(input_size):
                dWxh[h][i] += dh_raw[h] * inputs[t][i]
            for j in range(hidden_size):
                dWhh[h][j] += dh_raw[h] * hidden_states[t][j]
            dbh[h] += dh_raw[h]
        dh_next = dh_raw

    for h in range(hidden_size):
        for i in range(input_size):
            Wxh[h][i] -= learning_rate * dWxh[h][i]
        for j in range(hidden_size):
            Whh[h][j] -= learning_rate * dWhh[h][j]
        bh[h] -= learning_rate * dbh[h]

    for o in range(output_size):
        for h in range(hidden_size):
            Why[o][h] -= learning_rate * dWhy[o][h]
        by[o] -= learning_rate * dby[o]

print("\nFinal Prediction:")
predicted_index = output_probs.index(max(output_probs))
print(f"Input: {' '.join(input_words)}")
print(f"Predicted word: {index_to_word[predicted_index]}")

import numpy as np

def generate_sample_data(num_users=300, min_clicks=30, max_clicks=60, seed=42):
    np.random.seed(seed)
    data = []
    for user_id in range(num_users):
        user_type = np.random.choice(['morning', 'evening', 'bimodal'])
        num_clicks = np.random.randint(min_clicks, max_clicks)
        slots = []

        for _ in range(num_clicks):
            if user_type == 'morning':
                hour = int(np.clip(np.random.normal(8.5, 1.5), 6, 11))
            elif user_type == 'evening':
                hour = int(np.clip(np.random.normal(20, 1.5), 17, 23))
            else:
                if np.random.random() < 0.6:
                    hour = int(np.clip(np.random.normal(8.5, 1.5), 6, 11))
                else:
                    hour = int(np.clip(np.random.normal(20, 1.5), 17, 23))

            slot = hour * 12 + np.random.randint(0, 12)
            slots.append(slot)
        data.append(slots)
    return data


def prepare_train_data(user_sequences, seq_len):
    X, y = [], []
    for seq in user_sequences:
        for i in range(len(seq) - seq_len):
            X.append(seq[i:i+seq_len])
            y.append(seq[i+seq_len])
    return X, y


def get_data(seq_len):
    user_sequences = generate_sample_data(num_users=200)
    X, y = prepare_train_data(user_sequences, seq_len=seq_len)

    # 顺序随机打乱
    idx = np.random.permutation(len(X))
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:split + 300], y[split:split + 300]
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = get_data(seq_len=10)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
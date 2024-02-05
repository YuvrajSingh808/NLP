def tokenize_sentence(sentence, tokens_file):

    with open(tokens_file, 'r') as f:
        tokens = [line.strip() for line in f]

    tokens = sorted(tokens, key=len, reverse=True)

    tokenized_sentence = []

    i = 0
    while i < len(sentence):

        for token in tokens:
            if sentence[i:i+len(token)] == token:

                tokenized_sentence.append(token)
                i += len(token)
                break
        else:

            i += 1

    return tokenized_sentence


tokens_file = 'tokens.txt'
sentence = 'low the lower new, means widening the newer low in land'
tokenized_sample= tokenize_sentence(sentence, tokens_file)
with open('tokenized_samples.txt', 'w') as f:
    f.write(', '.join(tokenized_sample) + '\n')

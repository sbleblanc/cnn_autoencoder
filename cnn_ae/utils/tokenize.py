

def tokenize_to_char(string):
    char_string = []
    for w in string.split(' '):
        char_string.extend(list(w))
        char_string.append('<_>')
    if not string.endswith(' '):
        char_string.pop(-1)
    return char_string

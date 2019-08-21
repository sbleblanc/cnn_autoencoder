import re

def tokenize_to_char(string):
    char_string = []
    for w in string.split(' '):
        char_string.extend(list(w))
        char_string.append('<_>')
    if not string.endswith(' '):
        char_string.pop(-1)
    return char_string


class WordToCharTokenizer(object):

    def __init__(self, valid_content_regex=r"[a-z\s]+"):
        self.extract_re = re.compile(valid_content_regex)

    def __call__(self, string):
        char_string = []
        valid_parts = self.extract_re.findall(string.replace("\n", " "))
        for w in valid_parts:
            char_string.extend([c for c in w])
        char_string.append(' ')
        return char_string

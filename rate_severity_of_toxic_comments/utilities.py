__version__ = '0.1.0'
__author__ = 'Lorenzo Menghini, Martino Pulici, Alessandro Stockman, Luca Zucchini'

_bad_words = []

def obfuscator(text):
    global _bad_words
    if len(_bad_words) == 0:
        with open("res/bad_words.txt") as file:
            bad_words = [l.rstrip() for l in file.readlines() if len(l) > 2]
            bad_words = [l for l in bad_words if len(l) > 2]
    
        _bad_words = list(sorted(bad_words, key=len, reverse=True))

    for word in _bad_words:
        visible = min(len(word) // 3, 3)
        censorship = word[0:visible] + ((len(word) - visible * 2) * "*") + word[-visible:]
        text = text.replace(word, censorship)
    return text
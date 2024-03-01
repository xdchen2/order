'''
CFG Generating Sentences
'''

from nltk.parse.generate import generate
from nltk import CFG

# proper names
pn = CFG.fromstring("""
	S -> NP1 VT NP2
    NP1 -> 'Mary' | 'John' | 'Ben' | 'Jack' | 'Mike' | 'Linda' | 'Joe' | 'Amy' | 'Luke'
    NP2 -> 'Mary' | 'John' | 'Ben' | 'Jack' | 'Mike' | 'Linda' | 'Joe' | 'Amy' | 'Luke'
    VT -> 'marries' | 'talks to' | 'meets' | 'visits' | 'speaks to' | 'plays with' | 'likes'
""")

# general
gn = CFG.fromstring("""
	S -> NP1 VT NP2
    NP1 -> DET N1
    NP2 -> DET N2
    N1  -> 'chef' | 'police' | 'nurse' | 'teacher' | 'boy' | 'man' | 'woman' | 'worker' | 'girl'
    N2  -> 'onion' | 'beef' | 'butter' | 'orange' | 'banana' | 'apple' | 'fish' | 'chicken' | 'beef'
    VT -> 'eats' | 'likes' | 'dislikes' | 'sees' | 'looks at' | 'observes' | 'chops' | 'tastes'
    DET -> 'the'
""")

# multi-proper names
mpn = CFG.fromstring("""
	S -> NP1 VT NP1 | NP1 VT NP1 PP
    NP1 -> DET N1
    NP2 -> DET N2
    PP  -> P DET N3
    N1  -> 'fish' | 'book' | 'police' | 'writer' | 'bear' | 'stone'
    N3  -> 'restaurant' | 'river' | 'zoo' | 'office' | 'library'
    VT -> 'eats' | 'reads' | 'taps' | 'sees' | 'greets' | 'examines'
    DET -> 'the'
    P -> 'in'
""")

# multi-general
mgn = CFG.fromstring("""
	S -> NP1 VT NP1 | NP1 VT NP1 PP
    NP1 -> DET N1
    NP2 -> DET N2
    PP  -> P DET N3
    N1  -> 'fish' | 'book' | 'police' | 'writer' | 'bear' | 'stone'
    N3  -> 'restaurant' | 'river' | 'zoo' | 'office' | 'library'
    VT -> 'eats' | 'reads' | 'taps' | 'sees' | 'greets' | 'examines'
    DET -> 'the'
    P -> 'in'
""")


if __name__ == '__main__':

    save_path = '/Users/xdchen/Downloads/'

    sents = [pn, gn]

    sentn = ['pn', 'gn']

    for i, item in enumerate(sents):
        with open(save_path+sentn[i]+'.txt', 'w+') as f:
            for sentence in generate(item, depth=6):
                txt = ' '.join(sentence) + '\n'
                f.write(txt)
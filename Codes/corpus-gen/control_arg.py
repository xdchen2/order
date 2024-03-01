# check the position of the two arguements

import pandas as pd

def check_pos(sent, word_list):
    ''' input a single sent list
    out: list of permuted sent
    '''
    # idx can be more than one
    word_idx = []

    for i, item in enumerate(sent):
        if item in word_list:
            word_idx.append(i)

    return word_idx

def read(df_path):
    ''' read in a text as list
    three cols
    num, generated text, actual text
    '''
    with open(df_path) as file:
        next(file)
        fl = file.readlines()
        fl = [i.split(',') for i in fl]

    return fl


def compare(txt1, txt2):
    ''' take two sentences
    remove determiners
    '''
    txt1_new = []
    txt2_new = []

    for i in txt1:
        if i != 'the':
            txt1_new.append(i)

    for i in txt2:
        if i != 'the':
            txt2_new.append(i)

    if txt1_new == txt2_new:
        return 1
    else:
        return 0




def main():
    
    line_lists = read(path)

    scores = []

    for line in line_lists:
        src_sent = line[1].strip().split(' ')
        tgt_sent = line[2].strip().split(' ')

        score = compare(src_sent, tgt_sent)

        # if pn == 1:

        #     src_score1 = check_pos(src_sent, word_list_pn_sub)
        #     src_score2 = check_pos(src_sent, word_list_pn_obj)

        #     tgt_score1 = check_pos(tgt_sent, word_list_pn_sub)
        #     tgt_score2 = check_pos(tgt_sent, word_list_pn_obj)

        # if gn == 1:

        #     src_score1 = check_pos(src_sent, word_list_gn_sub)
        #     src_score2 = check_pos(src_sent, word_list_gn_obj)

        #     tgt_score1 = check_pos(tgt_sent, word_list_gn_sub)
        #     tgt_score2 = check_pos(tgt_sent, word_list_gn_obj)


    #     if src_score1 != tgt_score1 or src_score2 != tgt_score2:
    #         score = 0
    #     else:
    #         score = 1

        scores.append(score)

    print(sum(scores)/len(scores))


if __name__ == '__main__':

    # hyperparameters

    path = '/Users/xdchen/Downloads/proper_name_prd.csv'
    path = '/Users/xdchen/Downloads/standard_name_pred.csv'

    pn = 0
    gn = 1

    word_list_gn_sub = ['chef', 'police', 'nurse', 'teacher', 'boy', 'man', 'woman', 'worker', 'girl']
    word_list_gn_obj = ['onion', 'beef', 'butter', 'orange', 'banana', 'apple', 'fish', 'chicken', 'beef']
    word_list_pn_sub = ['Mary', 'John', 'Ben', 'Jack', 'Mike', 'Linda', 'Joe', 'Amy', 'Luke']
    word_list_pn_obj = ['Mary', 'John', 'Ben', 'Jack', 'Mike', 'Linda', 'Joe', 'Amy', 'Luke']

    main()

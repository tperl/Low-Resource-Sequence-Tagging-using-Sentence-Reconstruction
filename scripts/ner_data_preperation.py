import numpy as np
import re

def prepare_ner_data(src,dst,is_2003=False,is_dutch=False):
    with open(src, "r",encoding='utf-8' ) as f_in:
        mylist = f_in.read().splitlines()
        with open(dst, 'w',encoding='utf-8') as f_out:
            if is_2003:
                title = "Sentence #,Word,POS,CHUNK,Tag\n"
            else:
                title = "Sentence #,Word,POS,Tag\n"
            f_out.write(title)
            first_line = True

            count = 0
            sentence_counter = 1
            new_sentence = False
            for index, line in enumerate(mylist):
                if 'DOCSTART' in line or '===========' in line:
                    continue

                if first_line:
                    if line == '':
                        continue

                    if is_dutch:
                        if line[0] == ' ':
                            line = line[1:]
                    words = "Sentence: " + str(sentence_counter) + ',' + line.replace(' ', ',') + '\n'
                    f_out.write(words)
                    first_line = False
                    sentence_counter += 1
                    continue
                if line == '' or line == '   ':
                    new_sentence = True
                else:
                    # data cleaning...
                    num_strings = re.findall(r'[0-9][0-9,.]+', line)
                    for num in num_strings:
                        num_fixed = num.replace(',','.')
                        line = line.replace(num,num_fixed)

                    # r'\d+(?:,\d+)?'
                    suf_comma_words = re.findall("\w+,", line) + re.findall("\)+,", line)

                    if is_dutch:
                        if line[0] == ' ':
                            line = line[1:]
                        line = line.replace('","',',')

                    if len(suf_comma_words) == 1:
                        suf_comma_in_word = True
                        line = line.replace(suf_comma_words[0],suf_comma_words[0].split(',')[0])
                    else:
                        suf_comma_in_word = False

                    pre_comma_words = re.findall(",+\w", line)
                    if len(pre_comma_words) == 1:
                        pre_comma_in_word = True
                        line = line.replace(pre_comma_words[0],pre_comma_words[0].split(',')[1])
                    else:
                        pre_comma_in_word = False

                    if len(suf_comma_words) == 1:
                        suf_comma_in_word = True
                        line = line.replace(suf_comma_words[0], suf_comma_words[0].split(',')[0])
                    else:
                        suf_comma_in_word = False

                    split_line = line.split(' ')
                    dash_in_word = False
                    if '-' in split_line[0]:
                        dash_in_word = True

                    line = line.replace('"."','.')
                    line = line.replace(',', '^,^')

                    if index+1 != len(mylist):
                        if 'B-' in line and 'I-' not in mylist[index+1].split(' ')[-1]:
                            line = line.replace('B-', 'S-')
                        if 'I-' in line and 'I-' not in mylist[index+1].split(' ')[-1]:
                            line = line.replace('I-', 'E-')


                    if new_sentence or (count > 60 and 'I-' not in line and 'E-' not in line and '.' not in line):
                        words = "Sentence: " + str(sentence_counter)+',' + line.replace(' ',',')+'\n'
                        new_sentence = False
                        sentence_counter += 1
                        count = 0
                    else:
                        words = ',' + line.replace(' ', ',')+'\n'
                        count+= 1

                    if is_dutch:
                        words = words.replace(",,Punc,O,,",", ,Punc,O")

                    if suf_comma_in_word:
                        f_out.write(words)
                        comma_word = ',^,^,Fc,O\n'
                        f_out.write(comma_word)
                    elif pre_comma_in_word:
                        comma_word = ',^,^,Fc,O\n'
                        f_out.write(comma_word)
                        f_out.write(words)
                    else:
                        f_out.write(words)




def main():
    src_t = "../../data/Conll2003/eng/eng.train"
    dst_t = "../../data/Conll2003/eng/eng.train_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=True)
    # # prepare ConLL2003 validation data
    src_t = "../../data/Conll2003/eng/eng.testa"
    dst_t = "../../data/Conll2003/eng/eng.testa_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=True)
    # prepare ConLL2003 test data
    src_t = "../../data/Conll2003/eng/eng.testb"
    dst_t = "../../data/Conll2003/eng/eng.testb_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=True)

    src_t = "../../data/Conll2002\span\esp.train"
    dst_t = "../../data/Conll2002\span\esp.train_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=False)
    # prepare ConLL2003 validation data
    src_t = "../../data/Conll2002\span\esp.testa"
    dst_t = "../../data/Conll2002\span\esp.testa_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=False)
    # prepare ConLL2003 test data
    src_t = "../../data/Conll2002\span\esp.testb"
    dst_t = "../../data/Conll2002\span\esp.testb_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=False)

    src_t = "../../data/Conll2002/ned/ned.train"
    dst_t = "../../data/Conll2002/ned/ned.train_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=False, is_dutch=True)
    # prepare ConLL2003 validation data
    src_t = "../../data/Conll2002/ned/ned.testa"
    dst_t = "../../data/Conll2002/ned/ned.testa_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=False, is_dutch=True)
    # prepare ConLL2003 test data
    src_t = "../../data/Conll2002/ned/ned.testb"
    dst_t = "../../data/Conll2002/ned/ned.testb_updated.csv"
    prepare_ner_data(src_t, dst_t, is_2003=False, is_dutch=True)


if __name__ == "__main__":
    main()

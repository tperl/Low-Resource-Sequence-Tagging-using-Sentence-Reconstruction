

"""function expects to receive a list of words and a list of tags
 it then extract the entities and tags per entity for later to be use when comparing
 when comparing f1 scores"""
from builtins import enumerate
from seqeval.metrics import accuracy_score, f1_score, classification_report
import numpy as np


def extract_entites(tags):
    entity_tag = []
    is_entity = False
    for i in range(len(tags)):
        # print(tags[i])
        if "B-" in tags[i]:
            # if is_entity:
            #     entity_tag.append(tag)
            is_entity = True
            tag = []
            tag.append(tags[i])
            tag.append(i)
        elif "I-" in tags[i]:
            if i == 0:  # Wrong I- entity
                tag = []
            if not is_entity:
                # tag = []
                continue  # didnt get entity
            tag.append(tags[i])
            tag.append(i)
            is_entity = True
        elif "E-" in tags[i]:
            if not is_entity:
                continue
            tag.append(tags[i])
            tag.append(i)
            entity_tag.append(tag)
            is_entity = False
        elif "S-" in tags[i]:
            tag = []
            tag.append(tags[i])
            tag.append(i)
            entity_tag.append(tag)
            is_entity = False
        elif tags[i] == 'O':
            continue

        elif tags[i] == "END":
            return entity_tag
        else:
            print("error got unknown NER tag", tags[i])
            exit(1)

    return entity_tag


def testPOS(exp_tags, pred_tags, debug):
    F1_score= f1_score(exp_tags, pred_tags)
    acc_score= accuracy_score(exp_tags, pred_tags)
    cl_report= classification_report(exp_tags, pred_tags)
    print("F1 Score = ", F1_score)
    print("Accuracy Score = ", acc_score)
    print("Classification report = ", cl_report)
    return F1_score, acc_score, cl_report


""" 
expecting a list of tags and expected tags
"""
def f1_test(exp_tags, pred_tags, debug):
    Tp, Tn, Fp, Fn = 0, 0, 0, 0

    bad_sentence_counter = 0
    entity_false_counter = np.zeros((4))
    for i in range(len(exp_tags)):
        exp_entities = extract_entites(exp_tags[i])
        pred_entities = extract_entites(pred_tags[i])

        if debug:
            print('sample {0} GT= {1}' .format(i, exp_entities))
            print('sample {0} PRED= {1}' .format(i, pred_entities))
        was_fn = False
        was_fp = False
        for j, entity in enumerate(pred_entities):
            if entity in exp_entities:
                Tp += 1
            else:
                Fp += 1
                was_fp = True
                if 'B-PER' in entity:
                    entity_false_counter[0] += 1
                elif 'B-ORG' in entity:
                    entity_false_counter[1] += 1
                elif 'B-LOC' in entity:
                    entity_false_counter[2] += 1
                elif 'B-MISC' in entity:
                    entity_false_counter[3] += 1

        for j, entity in enumerate(exp_entities):
            if entity not in pred_entities:
                Fn += 1
                was_fn = True

        if was_fp and debug:
            print("There was a False Positive")
        if was_fn and debug:
            print("There was a False Negative")

        if was_fp or was_fn:
            bad_sentence_counter += 1
    if Tp==0:
        precision = 0
        recall = 0
        F1_score = 0
    else:
        precision = Tp /(Tp + Fp)
        recall = Tp / (Tp + Fn)
        F1_score = 2 / (1/precision + 1/recall)

    print("Summary")
    print("--------------")
    print("Tp = ", Tp)
    print("Fp = ", Fp)
    print("Fn = ", Fn)
    print("Num of bad sentence = ", bad_sentence_counter)
    print("Precision = ", precision)
    print("Recall = ", recall)
    print("F1 Score = ", F1_score)
    print("--------------")
    print('false PER entities: ',entity_false_counter[0])
    print('false ORG entities: ',entity_false_counter[1])
    print('false LOC entities: ',entity_false_counter[2])
    print('false MISC entities: ',entity_false_counter[3])
    return F1_score, precision, recall


if __name__ == "__main__":
    sentence = "President Barak Obama is the leader of the free world".split(" ")

    tags = "B-PER I-PER I-PER O O O O O O O".split(" ")
    tags2 = "B-PER I-PER I-PER O O O O O O O".split(" ")

    # compare_tokens(tags,tags2)
    tags3 = "O O O O O B-LOC I-LOC I-LOC O B-PER I-PER I-PER O O".split(" ")
    tags4 = "O O O O O O O".split(" ")

    tag5 = "B-ORG I-ORG I-ORG I-ORG I-ORG O B-PER I-PER O".split(" ")
    tag6 = "B-ORG I-ORG O I-MISC I-MISC O B-PER I-PER O".split(" ")
    # t = extract_entites(tag5)
    # t2 = extract_entites(tag6)

    exp_tag_list = [tags,tags,tags]
    tag_list = [tags2,tags2,tags2]
    # Tp, Tn, Fp, Fn = compare_entities(t, t2)
    f1 = f1_test(exp_tag_list, tag_list, True)

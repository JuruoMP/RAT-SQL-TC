import json
import os.path

from preprocess.process_turn_switch import process_dev as process_dev_turn_switch
from preprocess.process_tcs_column_table import process_dev as process_dev_col_tab


IS_REORDER = False

SEP = '<#>'


def format_one_session(session_dict, is_train):
    isfinal = False
    return_jsons = []

    database_id = session_dict["database_id"]
    interaction = session_dict["interaction"]
    for i in range(len(interaction)):
        query = interaction[i]["query"]
        if IS_REORDER:
            before_questions = [" ".join(turn['utterance_toks']) for turn in interaction[:i]]
            before_questions = [" ".join(interaction[i]["utterance_toks"])] + before_questions
        else:
            before_questions = [" ".join(turn['utterance_toks']) for turn in interaction[:i + 1]]

        question = SEP + " " + " {} ".format(SEP).join(before_questions)
        question_toks = question.split()
        sql = interaction[i]["sql"]

        return_json = {"db_id": database_id, "query": query, "question": question, "question_toks": question_toks,
                       "sql": sql}
        return_jsons.append(return_json)

    # +final
    for i in range(len(session_dict['interaction'])):
        if len(session_dict['interaction']) != 0:
            last_q = session_dict['interaction'][i]['query'].replace(' ', '').lower()
            final_q = session_dict['final']['query'].replace(' ', '').lower()
            if last_q[-1] == ";":
                last_q = last_q[0:-1]
            if final_q[-1] == ";":
                final_q = final_q[0:-1]
            if last_q == final_q:
                sql = interaction[i]["sql"]
                query = interaction[i]["query"]
                question = SEP + " " + session_dict['final']["utterance"]
                question_toks = question.split()
                return_json = {"db_id": database_id, "query": query, "question": question,
                               "question_toks": question_toks, "sql": sql}
                if is_train:
                    return_jsons.append(return_json)
                    isfinal = True
                    break
    return return_jsons, isfinal


def reformat(in_path, is_train):
    with open(in_path, 'r') as f:
        sparc = json.load(f)
    print("session_num:{}".format(len(sparc)))
    final_number = 0
    spider = []
    for session_dict in sparc:
        the_spider, is_final = format_one_session(session_dict, is_train)
        spider += the_spider
        if is_final:
            final_number += 1
    print("final add number:{}".format(final_number))

    max_query_len = 0
    for sample in spider:
        length = len(sample['question_toks'])
        if length > max_query_len:
            max_query_len = length
    print("max_query_len:{}".format(max_query_len))
    print("final_len:{}".format(len(spider)))

    return spider


def process_sparc_train():
    data_dict = reformat("raw_data/sparc/train.json", False)
    data_dict = process_dev_turn_switch(data_dict)
    data_dict = process_dev_col_tab(data_dict)
    os.makedirs('sparc_preproc', exist_ok=True)
    json.dump(data_dict, open('sparc_preproc/train.json', 'w'), ensure_ascii=False)


def process_sparc_dev():
    data_dict = reformat("raw_data/sparc/dev.json", False)
    data_dict = process_dev_turn_switch(data_dict)
    data_dict = process_dev_col_tab(data_dict)
    os.makedirs('sparc_preproc', exist_ok=True)
    json.dump(data_dict, open('sparc_preproc/dev.json', 'w'), ensure_ascii=False)


if __name__ == "__main__":
    process_sparc_train()
    process_sparc_dev()

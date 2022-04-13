import json
import re
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

label_counter_dict = defaultdict(int)

from seq2struct.utils.label_vocab import LabelVocabulary

#https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}
COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


def extract_select_clause(turn_):
    sql = turn_['sql']
    select = sql['select']
    is_distinct = select[0]
    select_agg_val_list = select[1]
    return select_agg_val_list, is_distinct


def get_select_change_list(select_agg_val_, select_agg_val_prev_, is_compare_col=False):
    change_list = []
    agg_op_0 = AGG_OPS[select_agg_val_[0]]
    if select_agg_val_[0] != select_agg_val_prev_[0]:
        if select_agg_val_[0] == 0:
            change_list.append("SELECT 改变了AGG:{}".format(AGG_OPS[select_agg_val_prev_[0]]))
        elif select_agg_val_prev_[1] == 0:
            change_list.append("SELECT 改变了AGG:{}".format(AGG_OPS[select_agg_val_[0]]))
        else:
            change_list.append("SELECT 改变了AGG:={}".format(agg_op_0))
    unit_op_0 = UNIT_OPS[select_agg_val_[1][0]]
    #if UNIT_OPS[select_agg_val_[1][0]] != UNIT_OPS[select_agg_val_prev_[1][0]]:
        #change_list.append("SELECT 改变了UNIT_OP:={}".format(unit_op_0))
        #return change_list
    agg_op_column_distinct = select_agg_val_[1][1]
    agg_op_column_distinct_prev = select_agg_val_prev_[1][1]

    agg_op_1 = agg_op_column_distinct[0]
    index_of_column_names = agg_op_column_distinct[1]
    is_distinct_1 = agg_op_column_distinct[2]  # [3, [0, [0, 5, False], None]]
    agg_op_1_prev = agg_op_column_distinct_prev[0]
    index_of_column_names_prev = agg_op_column_distinct_prev[1]
    is_distinct_1_prev = agg_op_column_distinct_prev[2]

    if is_distinct_1 != is_distinct_1_prev:
        change_list.append("SELECT 改变了DISTINCT:={}".format(is_distinct_1))

    if agg_op_1 != agg_op_1_prev:
        if agg_op_1 == 0:
            change_list.append("SELECT 改变了AGG:{}".format(AGG_OPS[agg_op_1_prev]))
        elif agg_op_1_prev == 0:
            change_list.append("SELECT 改变了AGG:{}".format(AGG_OPS[agg_op_1]))
        else:
            change_list.append("SELECT 改变了AGG:={}".format(AGG_OPS[agg_op_0]))

    if is_compare_col and index_of_column_names != index_of_column_names_prev:
        change_list.append("SELECT 改变了列:{}".format(index_of_column_names))
    return change_list


def is_cur_in_prev(select_agg_val, select_agg_val_list_prev):
    for i in range(len(select_agg_val_list_prev)):
        change_ = get_select_change_list(select_agg_val, select_agg_val_list_prev[i], True)
        if len(change_) == 0:
            return True
    return False


def extract_col_select(select_agg_val_):
    agg_op_0 = AGG_OPS[select_agg_val_[0]]
    unit_op_0 = UNIT_OPS[select_agg_val_[1][0]]

    # if unit_op_0 != 0:
    #     select_agg_val_[1]

    index_of_column_names_list = []
    for item in select_agg_val_[1:]:
        agg_op_column_distinct = item[1]
        agg_op_1 = agg_op_column_distinct[0]
        index_of_column_names = agg_op_column_distinct[1]
        is_distinct_1 = agg_op_column_distinct[2]  # [3, [0, [0, 5, False], None]]
        index_of_column_names_list.append(index_of_column_names)

    # agg_op_column_distinct = select_agg_val_[1][1]
    # agg_op_1 = agg_op_column_distinct[0]
    # is_distinct_1 = agg_op_column_distinct[2]  # [3, [0, [0, 5, False], None]]
    return agg_op_0, unit_op_0, index_of_column_names_list


def get_select_delete(select_agg_val_list, select_agg_val_list_prev):
    change_list = []
    for i in range(len(select_agg_val_list_prev)):
        if not is_cur_in_prev(select_agg_val_list_prev[i], select_agg_val_list):
            # 当前的被删除了
            agg0, _, index_of_column_names_list = extract_col_select(select_agg_val_list_prev[i])
            for index_of_column_names in index_of_column_names_list:
                if agg0 == 'none' and index_of_column_names == 0:
                    change_list.append("SELECT 删除了列:col={}".format(index_of_column_names))
                else:
                    change_list.append("SELECT 删除了列:col={}".format(index_of_column_names))
    return change_list


def get_select_add(select_agg_val_list, select_agg_val_list_prev):
    change_list = []
    for i in range(len(select_agg_val_list)):
        if not is_cur_in_prev(select_agg_val_list[i], select_agg_val_list_prev):
            # 当前增加了
            agg0, agg1, index_of_column_names_list = extract_col_select(select_agg_val_list[i])
            for index_of_column_names in index_of_column_names_list:
                if agg0 == 'none' and index_of_column_names == 0:
                    change_list.append("SELECT 增加了列:col={}".format(index_of_column_names))
                else:
                    change_list.append("SELECT 增加了列:col={}".format(index_of_column_names))

    return change_list


# 读取数据
def read_sparc_json(path="train_sparc_final_bak.json"):
    with open(path, 'r') as f:
        train_sparc_final = json.load(f)
    return train_sparc_final


def diff_select(turn, prev_turn):
    change_list = []
    select_agg_val_list, is_distinct = extract_select_clause(turn)
    select_agg_val_list_prev, is_distinct_prev = extract_select_clause(prev_turn)
    if is_distinct != is_distinct_prev:
        change_list.append("SELECT 改变了DISTINCT:={}".format(is_distinct))
    if len(select_agg_val_list) == len(select_agg_val_list_prev):
        # 找出变化的位置
        for select_agg_val, select_agg_val_prev in zip(select_agg_val_list, select_agg_val_list_prev):
            # print(select_agg_val)
            # [(agg_id, val_unit), (agg_id, val_unit), ...]
            change_list__ = get_select_change_list(select_agg_val, select_agg_val_prev)
            change_list.extend(change_list__)
    change_list_delete = get_select_delete(select_agg_val_list, select_agg_val_list_prev)
    change_list.extend(change_list_delete)
    change_list_add = get_select_add(select_agg_val_list, select_agg_val_list_prev)
    change_list.extend(change_list_add)
    return change_list


def extract_from_clause(turn_):
    sql = turn_['sql']
    from_ = sql['from']
    table_units = from_['table_units']
    conds = from_['conds']
    return table_units, conds


def is_from_table_in(item, table_units):
    for unit in table_units:
        if item[0] == unit[0] and item[1] == unit[1]:
            return True
    return False


def extract_from_change(table_units, table_units_prev):
    change_table = []

    table_set = set([str(item) for item in table_units])
    table_set_prev = set([str(item) for item in table_units_prev])
    if len(table_set.intersection(table_set_prev)) == len(table_set) == len(table_set_prev):
        return change_table  # 完全相同, 没有变化
    for item in table_units:
        if not is_from_table_in(item, table_units_prev):
            change_table.append("FROM 增加了表连接:{}".format(item))
    for item in table_units_prev:
        if not is_from_table_in(item, table_units):
            change_table.append("FROM 删除了表连接:{}".format(item))
    return change_table


def read_conds(cond):
    not_val = cond[0]
    where_ops_val = cond[1]
    clause_ = cond[2]
    unit_ops_val = UNIT_OPS[clause_[0]]
    col_op_clause = clause_[1]
    is_distinct = clause_[2]
    return not_val, where_ops_val, unit_ops_val, col_op_clause, is_distinct


def extract_Join_cond(conds, conds_prev):
    change_table = []
    for ii, cond in enumerate(conds):

        if cond == 'and':
            continue

        if ii > len(conds_prev) - 1:
            not_val = cond[0]
            where_ops_val = WHERE_OPS[cond[1]]
            clause_ = cond[2]
            unit_ops_val = UNIT_OPS[clause_[0]]
            col_op_clause = clause_[1]
            is_distinct = clause_[2]
            clause_ = cond[3]
            change_table.append("Join 增加了条件:{}{}{}".format(col_op_clause, where_ops_val, clause_))
            continue

        cond_prev = conds_prev[ii]

        not_val, where_ops_val, unit_ops_val, col_op_clause, is_distinct = read_conds(cond)
        not_val_prev, where_ops_val_prev, unit_ops_val_prev, col_op_clause_prev, is_distinct_prev = read_conds(
            cond_prev)

        if where_ops_val != where_ops_val_prev:
            change_table.append("Join 改变了WHERE_OPS:{}".format(WHERE_OPS[where_ops_val_prev]))
        if unit_ops_val != unit_ops_val_prev:
            change_table.append("Join 改变了UNIT_OPS:{}".format(UNIT_OPS[unit_ops_val_prev]))
        if col_op_clause[0] != col_op_clause_prev[0]:
            change_table.append("Join 改变了Join AGG_OPS:{}".format(AGG_OPS[col_op_clause_prev[0]]))
        if col_op_clause[1] != col_op_clause_prev[1]:
            pass
            #change_table.append("Join列的名称:{}".format(col_op_clause_prev[1]))

    return change_table


def diff_from_by_turn(turn, prev_turn):
    change_table = []

    table_units, conds = extract_from_clause(turn)
    table_units_prev, conds_prev = extract_from_clause(prev_turn)
    t = extract_from_change(table_units, table_units_prev)
    change_table.extend(t)
    # print(change_table)
    # ----以下找出连接的条件变化---
    if len(change_table) == 0:
        if len(conds) != 0:
            change_table.extend(extract_Join_cond(conds, conds_prev))
    return change_table


def extract_cond_where(cond):
    is_distinct = cond[0]
    agg_op_val = WHERE_OPS[cond[1]]
    unit_op_agg_val_ = cond[2]
    unit_op_val = UNIT_OPS[unit_op_agg_val_[0]]
    agg_col_value = unit_op_agg_val_[1]
    col_has_value = cond[3]
    return is_distinct, agg_op_val, unit_op_val, agg_col_value, col_has_value


def is_in_where_cond(cond, where_agg_val_list_prev):
    if cond in ("and", "or"):
        return cond in where_agg_val_list_prev
    is_distinct, agg_op_val, unit_op_val, agg_col_value, col_has_value = extract_cond_where(cond)

    for i in range(len(where_agg_val_list_prev)):
        cond_prev = where_agg_val_list_prev[i]
        if cond_prev in ("and", "or"):
            continue
        is_distinct_prev, agg_op_val_prev, unit_op_val_prev, agg_col_value_prev, col_has_value_prev = extract_cond_where(
            cond_prev)
        if (is_distinct, agg_op_val, unit_op_val, agg_col_value, col_has_value) == (
                is_distinct_prev, agg_op_val_prev, unit_op_val_prev, agg_col_value_prev, col_has_value_prev):
            return True
    return False


def get_where_delete(where_agg_val_list, where_agg_val_list_prev):
    change_list = []
    for i in range(len(where_agg_val_list_prev)):
        if not is_in_where_cond(where_agg_val_list_prev[i], where_agg_val_list):
            if where_agg_val_list_prev[i] in ('and', 'or'):
                # 当前是增加的条件
                change_list.append(
                    "WHERE 删除了条件: {}".format(where_agg_val_list_prev[i]))
            else:
                is_distinct, agg_op_val, unit_op_val, agg_col_value, col_has_value = extract_cond_where(
                    where_agg_val_list_prev[i])
                # 当前的被删除了
                change_list.append("WHERE 删除了条件:col={},{},value={}".format(agg_col_value[1], agg_op_val, col_has_value))
    return change_list


def get_where_add(where_agg_val_list, where_agg_val_list_prev):
    change_list = []
    for i in range(len(where_agg_val_list)):
        if not is_in_where_cond(where_agg_val_list[i], where_agg_val_list_prev):
            if where_agg_val_list[i] in ('and', 'or'):
                # 当前是增加的条件
                change_list.append(
                    "WHERE 增加了条件and/or:{}".format(where_agg_val_list[i]))
            else:
                is_distinct, agg_op_val, unit_op_val, agg_col_value, col_has_value = extract_cond_where(
                    where_agg_val_list[i])
                # 当前是增加的条件
                change_list.append(
                    "WHERE 增加了列条件:col={},{},value={}".format(agg_col_value[1], agg_op_val, col_has_value))
    return change_list


def diff_where_by_turn(turn, turn_prev):
    query = turn['query']

    sql = turn['sql']
    where_ = sql['where']

    sql_prev = turn_prev['sql']
    where_prev = sql_prev['where']

    change_list = []

    if len(where_) == 0:
        return change_list

    change_list.extend(get_where_delete(where_, where_prev))
    change_list.extend(get_where_add(where_, where_prev))

    return change_list


def is_in_groupby(col_unit, groupBy_list):
    for item in groupBy_list:
        if item == col_unit:
            return True
    return False

def is_in_orderby(col_unit, orderBy_list):
    for item in orderBy_list:
        for col_unit_iter in item:
            if col_unit_iter == col_unit:
                return True
    return False


def get_groupby_add(groupby_list, groupby_list_prev):
    change_list = []
    for i in range(len(groupby_list)):
        if not is_in_groupby(groupby_list[i], groupby_list_prev):
            change_list.append("groupby增加了列: {}".format(groupby_list[i][1]))
    return change_list


def get_groupby_delete(groupby_list, groupby_list_prev):
    change_list = []
    for i in range(len(groupby_list_prev)):
        if not is_in_groupby(groupby_list_prev[i], groupby_list):
            change_list.append("groupby删除了列: {}".format(groupby_list_prev[i][1]))
    return change_list

def get_orderby_add(by_list, by_list_prev):
    change_list = []
    if len(by_list) == 0: return change_list  # 无从增加和删除

    if by_list[0] not in by_list_prev:  # desc/asc 的比较
        val_unit_list = by_list[1]
        for val_unit in val_unit_list:
            change_list.append("orderBy改变了:{} {}".format(val_unit[1][1], by_list[0]))

    val_unit_list = by_list[1]
    for i in range(len(val_unit_list)):
        if not is_in_orderby(val_unit_list[i], by_list_prev[1:]):
            for val_unit in val_unit_list:
                change_list.append("orderBy改变了:{}".format(val_unit[1][1]))
    return change_list

def get_orderby_delete(by_list, by_list_prev):
    change_list = []
    for i in range(len(by_list_prev[1:])):
        if not is_in_groupby(by_list_prev[i], by_list):
            change_list.append("orderBy改变了: {}".format(by_list_prev[0]))
    return change_list


def diff_group_by_turn(turn, turn_prev):
    sql = turn['sql']
    groupBy = sql['groupBy']

    sql_prev = turn_prev['sql']
    groupBy_prev = sql_prev['groupBy']

    change_list = []

    if len(groupBy) == 0 and len(groupBy_prev) == 0:
        return change_list

    change_list.extend(get_groupby_add(groupBy, groupBy_prev))
    change_list.extend(get_groupby_delete(groupBy, groupBy_prev))

    return change_list

def diff_group_by_orderby(turn, turn_prev):
    sql = turn['sql']
    orderBy = sql['orderBy']

    sql_prev = turn_prev['sql']
    orderBy_prev = sql_prev['orderBy']

    change_list = []

    if len(orderBy) == 0 and len(orderBy_prev):
        return change_list

    change_list.extend(get_orderby_add(orderBy, orderBy_prev))
    change_list.extend(get_orderby_add(orderBy_prev, orderBy))

    return change_list


def diff_group_by_limit(turn, turn_prev):
    sql = turn['sql']
    limit = sql['limit']

    sql_prev = turn_prev['sql']
    limit_prev = sql_prev['limit']

    change_list = []
    if limit != limit_prev:
        change_list.append("改变了LIMIT:{}".format(limit))
    return change_list


def diff_group_by_intersect(turn, turn_prev):
    sql = turn['sql']
    intersect = sql['intersect']

    sql_prev = turn_prev['sql']
    intersect_prev = sql_prev['intersect']

    change_list = []

    if (intersect is None or intersect_prev is None) and (intersect or intersect_prev):
        change_list.append("改变了intersect:{}".format(intersect))

    if (turn['sql']['except'] is None or turn_prev['sql']['except']  is None) and (turn['sql']['except']  or turn_prev['sql']['except'] ):
        change_list.append("改变了except:{}".format(turn['sql']['except']))

    if (turn['sql']['union'] is None or turn_prev['sql']['union'] is None) and (
            turn['sql']['union'] or turn_prev['sql']['union']):
        change_list.append("改变了union:{}".format(turn['sql']['union']))

    return change_list



def process_turn_switch(train_sparc_final):
    for i, turn in enumerate(train_sparc_final):

        question = turn['question']
        seprators = re.findall("<#>", question)
        if len(seprators)>1:
            change_list_all = []
            prev_turn = train_sparc_final[i - 1]

            print(prev_turn['query'])
            print(turn['query'])

            change_list = diff_select(turn, prev_turn)
            change_list_from = diff_from_by_turn(turn, prev_turn)
            change_list_where = diff_where_by_turn(turn, prev_turn)
            change_list_groupby = diff_group_by_turn(turn, prev_turn)
            change_list_orderby = diff_group_by_orderby(turn, prev_turn)
            change_list_limit = diff_group_by_limit(turn, prev_turn)
            change_list_intersect = diff_group_by_intersect(turn, prev_turn)

            change_list.extend(change_list_from)
            change_list.extend(change_list_where)
            change_list.extend(change_list_groupby)
            change_list.extend(change_list_orderby)
            change_list.extend(change_list_limit)
            change_list.extend(change_list_intersect)



            # ---------增加一层逻辑处理--------
            change_list = process_oppsite_op(change_list)

            change_list_prev = prev_turn['turn_change']

            change_list_all.extend(change_list_prev)
            change_list_all.append(change_list)


            turn['turn_change'] = change_list_all

            print(turn['turn_change'][-1])
            print("==============")

        else:
            turn['turn_change'] = [[]]  # 第一轮

    return train_sparc_final


def process_oppsite_op(change_list):
    ll = []
    col_op_dict = dict()
    for change_ in change_list:
        re_o = re.match("SELECT \w\w了列:col=(\d+)", change_)
        if re_o:
            col_ = re_o.group(1)
            if col_ in col_op_dict:
                col_op_dict[col_] = True
            else:
                col_op_dict[col_] = False
    for change_ in change_list:
        re_o = re.match("SELECT \w\w了列:col=(\d+)", change_)
        if re_o and col_op_dict[re_o.group(1)]:
            continue
        ll.append(change_)
    return ll


def main():
    vocab = LabelVocabulary()

    train_sparc_final = read_sparc_json("train_sparc_final_bak.json")
    process_turn_switch(train_sparc_final)

    for turn in train_sparc_final:
        turn_change = turn['turn_change']

        change_index_list = [[vocab.add_token_to_namespace(c.split(":")[0], "turn_switch") for c in change_list] for
                             change_list in turn_change]
        turn['turn_change_index'] = change_index_list

        for index_l in change_index_list:
            for item in index_l:
                label_counter_dict[item] += 1

    dev_sparc_final = read_sparc_json("dev_sparc.json")
    process_turn_switch(dev_sparc_final)

    for turn in dev_sparc_final:
        turn_change = turn['turn_change']
        for change_list in turn_change:

            change_index_list = [[vocab.add_token_to_namespace(c.split(":")[0], "turn_switch") for c in change_list] for
                                 change_list in turn_change]
            turn['turn_change_index'] = change_index_list

            for index_l in change_index_list:
                for item in index_l:
                    label_counter_dict[item] += 1

    index_token_obj = vocab.get_index_to_token_vocabulary()

    # op_list = list(index_token_obj.values())
    # op_list.sort()
    # index_token_obj_new = {}
    # for v in op_list:
    #     index_token_obj_new[len(index_token_obj_new)] = v

    print(label_counter_dict)
    for k, v in label_counter_dict.items():
        print("{} : {}".format(index_token_obj[k], v))

    # 写入 JSON 数据
    with open('train_sparc_final_op.json', 'w') as f:
        json.dump(train_sparc_final, f, ensure_ascii=False)
    # 写入 JSON 数据
    with open('dev_sparc_final_op.json', 'w') as f:
        json.dump(dev_sparc_final, f, ensure_ascii=False)

    # 写入 JSON 数据
    with open('turn_switch_label_vocab.json', 'w') as f:
        json.dump(index_token_obj, f, ensure_ascii=False)

    #vocab.save_to_files(directory="turn_switch_vocab")


def process_dev(dev_sparc_final):
    vocab_dict = json.load(open('static/turn_switch_label_vocab.json'))
    vocab = LabelVocabulary()
    vocab.set_by_index_to_token(vocab_dict)
    process_turn_switch(dev_sparc_final)

    for turn in dev_sparc_final:
        turn_change = turn['turn_change']
        for change_list in turn_change:

            change_index_list = [[vocab._token_to_index.get(c.split(":")[0], 0) for c in change_list] for
                                 change_list in turn_change]
            turn['turn_change_index'] = change_index_list

            for index_l in change_index_list:
                for item in index_l:
                    label_counter_dict[item] += 1

    return dev_sparc_final


if __name__ == '__main__':
    main()

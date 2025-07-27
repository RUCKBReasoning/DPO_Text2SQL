import sys
import sqlite3
import json
import argparse
import os
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import multiprocessing as mp
import re
from distutils.util import strtobool

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type = str, default = "./results/1b_sampling_dev_results.json")
    parser.add_argument('--gold', type = str, default = "./bird/dev/dev.json")
    parser.add_argument('--db_path', type = str, default = "./bird/dev/dev_databases")
    parser.add_argument('--output', type = str, default = "./results/1b_sampling_evaluation_dev_results.json")
    parser.add_argument('--is_cot', type = str, default="True") # the added flag to specify cot model
    
    opt = parser.parse_args()

    return opt

def extract_sql_from_cot(text, is_cot = True):
    # Extract all SQL code blocks from the generated text and return the last one
    if is_cot != True :
        return text 
    matches = re.findall(r"```(?i:sql)*\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return the last matched SQL
    matches = re.findall(r"`(.*?)`", text, re.DOTALL) # Inline code

    if matches:
        return matches[-1].strip()  # Return the last matched SQL
    return text # else is not cot model

def compare_sql(question_id, db_id, question, ground_truth, pred_sql) :
    conn = sqlite3.connect(os.path.join(opt.db_path, db_id, db_id + ".sqlite"))
    cursor = conn.cursor()
    try:
        cursor.execute(pred_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        if set(predicted_res) == set(ground_truth_res):
            res = 1
    except:
        res = 0
    conn.close()
    return question_id, db_id, question, ground_truth, pred_sql, res

def sql_execution_result(question_id, db_id, question, pred_sql) :
    conn = sqlite3.connect(os.path.join(opt.db_path, db_id, db_id + ".sqlite"))
    cursor = conn.cursor()
    try:
        cursor.execute(pred_sql)
        predicted_res = cursor.fetchall()
        predicted_res = frozenset(predicted_res)
        res = 1 # this sql is executable
    except:
        predicted_res = None
        res = 0
    conn.close()
    return question_id, db_id, question, pred_sql, predicted_res, res

def execute_wrapper(args, timeout) :
    '''Wrap execute_sql for timeout'''
    try:
        res = func_timeout(timeout, sql_execution_result, args=args)
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = (*args, None, 0)
    except Exception as e:
        res = (*args, None, 0)
    return res


def execute_callback(result, result_collection):
    '''Store the execution result in the collection'''
    question_id, db_id, question, pred_sql, predicted_res, res = result

    #print(result)
    
    if res == 0 :
        return

    result_collection.append( (pred_sql, predicted_res) )

def execute_sqls_parallel(db_ids, questions, all_grouped_pred_sqls, ground_truth_sqls, num_cpus=1, timeout=1):
    '''Execute each questions sqls in parallel'''
    #cnt = 0
    for question_id, db_id, question, grouped_pred_sqls, ground_truth in tqdm(zip([x for x in range(len(db_ids))], db_ids, questions, all_grouped_pred_sqls, ground_truth_sqls)):
        #cnt += 1 
        #if cnt == 5 :
        #    break
        pool = mp.Pool(processes=num_cpus)
        extract_grouped_pred_sqls = [extract_sql_from_cot(text, bool(strtobool(opt.is_cot))) for text in grouped_pred_sqls] # extract sql from cot
        selected_sql = None # final selected sql
        result_collection = [] # (pred_sql, predicted_res)
        major_voting_counting = {} # counting execution results
        for pred_sql in extract_grouped_pred_sqls:
            pool.apply_async(execute_wrapper, args=((question_id, db_id, question, pred_sql), timeout), callback=lambda res: execute_callback(res, result_collection))
        pool.close()
        pool.join()

        if result_collection == [] :
            # if no sql successfully executed
            selected_sql = extract_grouped_pred_sqls[0] # select a random one to return
            print(f'No successful execution')
            # store this majority sql as a inferenece result 
            bird_results_dict[question_id] = selected_sql + "\t----- bird -----\t" + db_id # sequential added, no need to resort
            continue 

        # process major voting
        for pred_sql, predicted_res in result_collection :
            if major_voting_counting.get(predicted_res) != None :
                major_voting_counting[predicted_res][0] += 1 # counting add one 
            else :
                major_voting_counting[predicted_res] = [1, pred_sql] # count and record sql
        # select majority voting
        major_vote = max(major_voting_counting.values(), key=lambda x: x[0])
        selected_sql = major_vote[-1]
        # print votes
        print(f'majority votes: {major_vote[0]}')
        # store this majority sql as a inferenece result 
        bird_results_dict[question_id] = selected_sql + "\t----- bird -----\t" + db_id # sequential added, no need to resort


if __name__ == "__main__":
    opt = parse_option()
    
    all_grouped_pred_sqls = json.load(open(opt.pred))
    db_ids = [data["db_id"] for data in json.load(open(opt.gold))]
    questions = [data["question"] for data in json.load(open(opt.gold))]
    ground_truth_sqls = [data["SQL"] for data in json.load(open(opt.gold))]

    print(len(all_grouped_pred_sqls), len(db_ids), len(questions), len(ground_truth_sqls))
    
    assert len(all_grouped_pred_sqls) == len(db_ids) == len(questions) == len(ground_truth_sqls)

    bird_results_dict = {}

    execute_sqls_parallel(db_ids, questions, all_grouped_pred_sqls, ground_truth_sqls, num_cpus=16, timeout=2)
    with open( opt.output, "w", encoding = 'utf-8') as f:
        f.write(json.dumps(bird_results_dict, indent = 2, ensure_ascii = False))
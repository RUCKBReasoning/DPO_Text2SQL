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

def execute_wrapper(args, timeout) :
    '''Wrap execute_sql for timeout'''
    try:
        res = func_timeout(timeout, compare_sql, args=args)
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = (*args, 0)
    except Exception as e:
        res = (*args, 0)
    return res


def execute_callback(result):
    '''Store the execution result in the collection'''
    question_id, db_id, question, ground_truth, pred_sql, res = result
    evaluation_res = dict()
    evaluation_res['question_id'] = question_id
    evaluation_res["db_id"] = db_id
    evaluation_res["question"] = question
    evaluation_res["ground_truth"] = ground_truth
    #evaluation_res["pred_sql"] = pred_sql
    #evaluation_res["correctness"] = res

    if question_id not in evaluation_results:
        evaluation_results[question_id] = evaluation_res
    
    ex_dicts[question_id].append((pred_sql, res))

    #only print in verbose mode
    #print('Done:', question_id, res) # Print the progress
    sys.stdout.flush()
    sys.stderr.flush()

def execute_sqls_parallel(db_ids, questions, all_grouped_pred_sqls, ground_truth_sqls, num_cpus=1, timeout=1):
    '''Execute the sqls in parallel'''
    pool = mp.Pool(processes=num_cpus)
    for question_id, db_id, question, grouped_pred_sqls, ground_truth in zip([x for x in range(len(db_ids))], db_ids, questions, all_grouped_pred_sqls, ground_truth_sqls):
        extract_grouped_pred_sqls = [extract_sql_from_cot(text, bool(strtobool(opt.is_cot))) for text in grouped_pred_sqls] # extract sql from cot
        for pred_sql in extract_grouped_pred_sqls:
            pool.apply_async(execute_wrapper, args=((question_id, db_id, question, ground_truth, pred_sql), timeout), callback=execute_callback)
    pool.close()
    pool.join()

if __name__ == "__main__":
    opt = parse_option()
    #print(opt)
    #print(bool(strtobool(opt.is_cot)))
    #exit()
    
    all_grouped_pred_sqls = json.load(open(opt.pred))
    db_ids = [data["db_id"] for data in json.load(open(opt.gold))]
    questions = [data["question"] for data in json.load(open(opt.gold))]
    ground_truth_sqls = [data["SQL"] for data in json.load(open(opt.gold))]

    print(len(all_grouped_pred_sqls), len(db_ids), len(questions), len(ground_truth_sqls))
    
    assert len(all_grouped_pred_sqls) == len(db_ids) == len(questions) == len(ground_truth_sqls)

    ex_dicts = {question_id: [] for question_id in range(len(questions))} # question_id -> [(pred_sql, res)]
    evaluation_results = {} # the missing ex_scores is filled at last, question_id -> dict
    
    execute_sqls_parallel(db_ids, questions, all_grouped_pred_sqls, ground_truth_sqls, num_cpus=20, timeout=20)

    for question_id in evaluation_results:
        ex_scores = []
        pred_sqls = []
        for pred_sql, res in ex_dicts[question_id]:
            ex_scores.append(res)
            pred_sqls.append(pred_sql)
        evaluation_results[question_id]["pred_sqls"] = pred_sqls
        evaluation_results[question_id]["ex_scores"] = ex_scores
    
    # sort evaluation_results with key
    evaluation_results = sorted(evaluation_results.items(), key=lambda x: x[0])

    # extract the values
    evaluation_results = [value for key, value in evaluation_results]
    
    with open(opt.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(evaluation_results, indent=2, ensure_ascii=False))
    
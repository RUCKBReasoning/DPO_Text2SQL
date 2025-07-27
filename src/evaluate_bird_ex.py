import sys
import sqlite3
import json
import argparse
import os
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm
import multiprocessing as mp

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type = str, default = "predict_dev.json")
    parser.add_argument('--gold', type = str, default = "./bird/dev/dev.json")
    parser.add_argument('--db_path', type = str, default = "./bird/dev/dev_databases")
    parser.add_argument('--output', type = str, default = "./results/1b_evaluation_dev_results.json")
    parser.add_argument('--is_cot', type = str, default="True") # the added flag to specify cot model
    
    opt = parser.parse_args()

    return opt

def compare_sql(question_id, db_id, question, ground_truth, pred_sql) :
    conn = sqlite3.connect(os.path.join(opt.db_path, db_id, db_id + ".sqlite"))
    cursor = conn.cursor()
    try:
        cursor.execute(pred_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        print('Successfully executed')
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
    evaluation_res["pred_sql"] = pred_sql
    evaluation_res["correctness"] = res
    evaluation_results.append(evaluation_res)
    results.append(res)

    print('Done:', question_id, res) # Print the progress
    sys.stdout.flush()
    sys.stderr.flush()

def execute_sqls_parallel(db_ids, questions, pred_sqls, ground_truth_sqls, num_cpus=1, timeout=1):
    '''Execute the sqls in parallel'''
    pool = mp.Pool(processes=num_cpus)
    for question_id, db_id, question, pred_sql, ground_truth in zip([x for x in range(len(db_ids))], db_ids, questions, pred_sqls, ground_truth_sqls):
        pool.apply_async(execute_wrapper, args=((question_id, db_id, question, ground_truth, pred_sql), timeout), callback=execute_callback)
    pool.close()
    pool.join()

if __name__ == "__main__":
    opt = parse_option()
    pred_results = json.load(open(opt.pred))
    pred_results = [pred_results[key] for key in pred_results]
    pred_sqls = [res.split("\t----- bird -----\t")[0] for res in pred_results]
    db_ids = [res.split("\t----- bird -----\t")[1] for res in pred_results]
    questions = [data["question"] for data in json.load(open(opt.gold))]
    ground_truth_sqls = [data["SQL"] for data in json.load(open(opt.gold))]
    
    assert len(pred_results) == len(pred_sqls) == len(db_ids) == len(questions) == len(ground_truth_sqls)

    evaluation_results = []
    results = []

    execute_sqls_parallel(db_ids, questions, pred_sqls, ground_truth_sqls, num_cpus=20, timeout=20)

    # sort evaluation_results by question_id
    evaluation_results = sorted(evaluation_results, key=lambda x:x['question_id'])
    
    print("EX:", sum(results)/len(results))
    with open(opt.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(evaluation_results, indent=2, ensure_ascii=False))
    
    # print accuracy
    print("Accuracy:", sum(results)/len(results))
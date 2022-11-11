import subprocess
import numpy as np
import json
import os

############### JAVA HELPERS #################
def compile_java(java_file):
    subprocess.check_call(['javac', #"-J-Xmx200m", "-J-Xms50m",
                           #"-J-XX:MaxPermSize=200m", "-J-XX:ReservedCodeCacheSize=10m",
                           #"-J-XX:-UseCompressedClassPointers",
                           java_file])

def execute_java(java_file, inputs):
    java_class,ext = os.path.splitext(java_file)
    cmd = ['java', "-Xmx50m", "-Xms50m", java_class]
    for inp in inputs:
        cmd.append(inp)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout


############## DATA GENERATOR ################
def generate_student_data():
    #read the whole data:
    original_data = np.genfromtxt("cal_housing.csv", delimiter=",", skip_header=1).astype(int)
    np.random.shuffle(original_data)
    
    #making decision data:
    decision = np.full((len(original_data)), False)
    while (np.sum(decision)<0.2 * len(decision)) or (np.sum(decision)>0.8 * len(decision)):
        dec_num = np.random.randint(2, 5, 1)[0]
        dec_features = np.random.choice(np.arange(0, original_data.shape[1]), size=dec_num, replace=False)
        decision = np.full((len(original_data)), True)
        for feature in dec_features:
            limit = np.random.choice(np.unique(original_data[:, feature]), size=1)[0]
            decision = np.logical_and(decision, original_data[:, feature]>limit)
            
    #saving resulted files:
    created_eval_data = np.hstack([original_data.astype(int), decision.reshape((len(decision),1))])
    #np.savetxt("evaluate.csv", created_eval_data, delimiter = ',', fmt="%d")
    np.savetxt("train.csv", created_eval_data[:150], delimiter = ",", fmt="%d")
    np.savetxt("test.csv", created_eval_data[:, :-1], delimiter=",", fmt="%d")
    return created_eval_data
    
    
############### EVALUATORS #################
def get_entropy(n_above, n_below):
    if n_above == 0 or n_below == 0: return 0.0
    p_above = n_above/(n_above+n_below)
    p_below = n_below/(n_above+n_below)
    return -(p_above*np.log2(p_above))-(p_below*np.log2(p_below))

def evaluate_entropy():
    #generate some random data:
    n_above = np.random.randint(1, 100)
    n_below = np.random.randint(1, 100)
    gt_entropy = get_entropy(n_above, n_below)
    #run the test scripts:
    compile_java('EvalEntropy.java')
    entropy_result = float(execute_java('EvalEntropy.java', [str(n_above), str(n_below)]))
    return np.abs(gt_entropy-entropy_result)<1e-5

def evaluate_separator(gt):
    def _eval_sep(features, labels, separation):
        features = np.array(features)
        labels = np.array(labels).reshape((len(labels), 1))
        data = np.hstack([features, labels])

        gains = {}
        best_gain = 0.0

        parent_entropy = get_entropy(np.sum(labels), len(labels)-np.sum(labels))    
        for j in range(features.shape[1]):
            uniques = np.unique(features[:, j])
            for u in uniques:
                sepa = "%d@%d"%(j,u)
                #print(sepa)
                split = np.logical_and(data.T[-1],data.T[1]<=u)
                #print(split)
                entropy = get_entropy(np.sum(split), len(split)-np.sum(split))
                gain = parent_entropy-entropy
                gains[sepa] = gain
                if gain>best_gain:
                    best_gain = gain

        return not(best_gain > gains[separation])

    #get some samples from the ground truth
    np.random.shuffle(gt)
    eval_df = gt[:100]
    features = eval_df[:, :-1]
    features_f = features.flatten()
    labels = eval_df[:, -1]
    
    #run the test scripts:
    compile_java('EvalSeparation.java')
    command = [str(features.shape[0]),
               str(features.shape[1])]
    for f in features_f:
        command.append(str(f))
    for l in labels:
        command.append('0' if l==False else '1')
    sep_result = execute_java('EvalSeparation.java', command)[:-1]
    
    #evaluate resulted separation:
    return _eval_sep(features, labels, sep_result)


def evaluate_dtree(gt):
    def _is_fake(results):
        '''We consider fake those solutions which contains only true or false values'''
        results = results==1
        return np.all(results) or np.all(~np.array(results))

    def _precision(gt, result):
        tp = np.sum(np.logical_and((decision == result), decision == True))
        fp = np.sum(np.logical_and((decision != result), results == True))
        return tp/(tp+fp) if tp+fp>0 else 0

    def _recall(gt, result):
        tp = np.sum(np.logical_and((decision == result), decision == True))
        fn = np.sum(np.logical_and((decision != result), results == False))
        return tp/(tp+fn) if tp+fn>0 else 0

    def _f2_score(gt, result):
        prec = _precision(gt, result)
        reca = _recall(gt, result)
        return 2*(prec*reca)/(prec+reca) if prec+reca > 0 else 0
        
    execute_java("Solution.java", [])
    #read results:
    #gt = np.genfromtxt("evaluate.csv", delimiter=",").astype(int)
    results = np.genfromtxt("results.csv", delimiter=",").astype(int)
    decision = gt[:, -1]

    #evaluation:
    if _is_fake(results):
        return 0
    if len(results) != len(decision):
        #print("wrong size")
        return 0
    #print(_f2_score(gt, results))
    return int(min(9, _f2_score(gt, results)*10))

def evaluate_hw():
    #compile the solution:
    compile_java('Solution.java')
    score = 0.0
    #create test data:
    gt = generate_student_data()
    try:
        if evaluate_entropy():
            score += 1.0/12.0
            #print("entropy ok")
    except:
        pass
    
    try:
        if evaluate_separator(gt):
            score += 2.0/12.0
            #print("separator ok")
    #eval 5 times accuracies:
    except:
        pass
    
    #we do not want to have fake solutions:
    if score<=0:
        return 0
    
    dtree_score = []
    try:
        for i in range(5):
            gt=generate_student_data()
            dtree_score.append(evaluate_dtree(gt))
        score += max(dtree_score)/12
    except:
        pass
    #print(dtree_score)
    return score
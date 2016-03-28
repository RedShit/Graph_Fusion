import re
#from ImageColor import str2int

def Evaluate(fn_label, fn_result, retri_amount = 20):

    print "Evaluate"

    label = {}

    fd_stdin = open(fn_label)

    num_img = 0
    for line in fd_stdin:
        line = line.rstrip()
        line = line.split()
        label[num_img] = line[-1]
        num_img += 1
    fd_stdin.close()

    fd_stdin = open(fn_result)

    precision = [0]*(retri_amount+1)
    ap = [0]*(retri_amount+1)
    NS_score = 0
    num_img = 0

    for line in fd_stdin:

        line = line.rstrip()
        line = line.split()

        idx = int(line[1])
        true_label = label[idx]
        cur_label = []
        for i in range(1, retri_amount+1):
            idx = int(line[i])
            cur_label.append(label[idx])

        NS_score += cur_label[0:4].count(true_label)

        for i in range(2,retri_amount+1):
            amount = cur_label[1:i].count(true_label)
            ap[i] += (amount+0.0)/(i-1.0)

        num_img += 1

        for i in range(0,retri_amount+1):
            amount = cur_label[0:i+1].count(true_label)
            precision[i] += float(amount) / (i + 1)


    if re.search('corel', fn_result):
        print "Precision:"
        for i in range(0, retri_amount):
            precision[i] = float(precision[i]) / (num_img)
            print i+1, precision[i]

    if re.search('uk', fn_result):
        print "NS score:"
        print NS_score
        print num_img
        NS_score = float(NS_score) / num_img
        print NS_score

    if re.search('oxford', fn_result):
        print "mAP"
        for i in range(2, retri_amount):
            ap[i] = float(ap[i]) / (num_img)
            print i-1, ap[i]

    fd_stdin.close()

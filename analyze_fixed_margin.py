# encoding=utf-8
import csv
import matplotlib.pyplot as plt
from pylab import *
import numpy as np


# csv_file="/home/goerlab/Welder_detection/data_record/20171114_Tue/cro-val3/unbalanced/vgg_16_ft+sensi+l2/result_detail2/detail_result.csv"


# margin=0
def calcute(csv_file, margin_preok, margin_preng, writer, mode):
    csv_reader = csv.reader(open(csv_file))

    total = csv_reader.line_num

    print("total:%d" % (total))

    pre_OK_correct_num = 0
    pre_NG_correct_num = 0
    miss_num = 0
    false_num = 0

    predict_OK = 0
    prediction_OK_wrong = 0

    predict_NG = 0
    prediction_NG_wrong = 0

    prob_pred_NG_to_OK = 0

    label_OK = 0
    label_NG = 0

    count = 0

    for row in csv_reader:
        if csv_reader.line_num == 1:
            continue
        else:
            fake_prediction = 0
            prediction = int(row[1])

            label = int(row[0])
            if label == 0:
                label_OK += 1
            else:
                label_NG += 1
            proba = float(row[2])

            if prediction == 0:
                if proba >= margin_preok:
                    fake_prediction = 0
                    predict_OK += 1
                else:
                    fake_prediction = 1
                    predict_NG += 1
            else:  ####prediction>1
                if proba >= margin_preng:
                    fake_prediction = 1
                    predict_NG += 1
                else:
                    fake_prediction = 0
                    predict_OK += 1

            if label == 0 and fake_prediction == 1:
                false_num += 1
            if label == 0 and fake_prediction == 0:
                pre_OK_correct_num += 1

            if label == 1 and fake_prediction == 0:
                miss_num += 1
            if label == 1 and fake_prediction == 1:
                pre_NG_correct_num += 1

    if label_NG == 0:
        label_NG = 1
    miss_rate = 1.0 * miss_num / label_NG
    if label_OK == 0:
        label_OK = 1
    false_rate = 1.0 * false_num / label_OK

    accuracy = 1.0 * (pre_OK_correct_num + pre_NG_correct_num) / (label_OK + label_NG)

    print("label OK:%d, label NG:%d, predict OK:%d, predict NG:%d" % (label_OK, label_NG, predict_OK, predict_NG))
    print("miss rate:%f,false rate:%f,accuracy:%f" % (miss_rate, false_rate, accuracy))

    return miss_rate, false_rate


if __name__ == "__main__":
    dirc_name = "/home/goerlab/Welder_detection/data_record/20180126/Bilinear/20180126_cross1_gan/"

    csv_file = dirc_name + "/test_detail_0126.csv"
    write_file = file(dirc_name + "/margin_analysis_miss_false_5.csv", "wb")
    # write_file="margin_analysis.csv"
    writer = csv.writer(write_file)
    margin = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1]
    # margin = np.linspace(0.3,0.99996,num=400)
    miss_rate_all = []

    false_rate_all = []


    margin_store = []
    fixed_i=0.99
    fixed_j=0.0
    # for i in margin:
    #     for j in margin:
    #         margin_store.append([i, j])
    #         print("margin:%d,%d" % (i, j))
    #         # miss_rate, false_rate ,average_prob= calcute(csv_file, i,writer)
    miss_rate, false_rate = calcute(csv_file, fixed_i, fixed_j, writer, "miss")
    print("miss_rate,false_rate:%f,%f" %(miss_rate,false_rate))

    #         miss_rate_all.append(miss_rate)
    #         false_rate_all.append(false_rate)
    #
    # # miss_rate_min_pos=miss_rate_all.index(min(miss_rate_all))
    # min_pos = []
    # for i in range(len(miss_rate_all)):
    #     if miss_rate_all[i] == min(miss_rate_all):
    #         min_pos.append(i)
    #
    # print("min of miss: %f" % (min(miss_rate_all)))
    # print("index:")
    # print(min_pos)
    #
    # min_false = 1
    # store_j = 0
    # for j in min_pos:
    #     print(miss_rate_all[j])
    #     if false_rate_all[j] < min_false:
    #         min_false=false_rate_all[j]
    #         store_j = j
    #
    # print("pos j:%d" % (j))
    # print("min miss,min false:%f,%f" % (miss_rate_all[store_j], false_rate_all[store_j]))
    # print("margin:")
    # print(margin_store[store_j])
    #
    # plt.plot(false_rate_all, miss_rate_all, "r.-", label=u"miss-rate")
    # plt.legend(loc='upper right')
    # plt.savefig(dirc_name + "/miss_false_rate4.png")
    # plt.show()
    #


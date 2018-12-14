from keras_retinanet.bin.train import train_main
from keras_retinanet import models
import glob
import numpy as np
from sklearn.metrics import f1_score
import os


if __name__ == '__main__':
    os.chdir("../")
    model_name = 'resnet101'
    train_main(0, None, ["csv", "data/trn1.csv", "data/classes.csv",
                       "--val-annotations", "data/val1.csv"])
    # model_name = 'resnet50'
    # h5s = glob.glob("run1_resnet_50/*.h5")
    # results = []
    # best_f1 = 0
    # for h5 in h5s:
    #     y_true, y_pred = train_main(1, h5, ["csv", "data/trn1.csv", "data/classes.csv",
    #                 "--val-annotations", "data/val1.csv"])
    #     f1_max = 0
    #     th_max = 0
    #     pr_cnt_max = 0
    #     for th in np.linspace(0.0, 1.0, num=21):
    #         for pr_cnt in range(1, 7):
    #             y_pred_new = []
    #             for prd in y_pred:
    #                 ref_p = prd[(np.argsort(prd))[-pr_cnt]]
    #                 dec = (prd >= ref_p) & (prd >= th)
    #                 y_pred_new.append(dec)
    #             f1_cur = f1_score(y_true, np.array(y_pred_new, dtype='int'), average='macro')
    #             if f1_cur >= f1_max:
    #                 f1_max = f1_cur
    #                 th_max = th
    #                 pr_cnt_max = pr_cnt
    #     results.append((h5, th_max, pr_cnt_max, f1_max))
    #     print([h5, th_max, pr_cnt_max, f1_max])
    #     if f1_max >= best_f1:
    #         best_f1 = f1_max
    #     print("current best = ", best_f1)
    #
    # results = sorted(results, key=lambda x:x[-1], reverse=True)
    # for r in results:
    #     print(r)

#[('snapshots\\resnet50_csv_44.h5', 0.05, 0.44536340852130324), ('snapshots\\resnet50_csv_48.h5', 0.05, 0.445054945054945), ('snapshots\\resnet50_csv_34.h5', 0.05, 0.437181855500821), ('snapshots\\resnet50_csv_49.h5', 0.0, 0.4327235488525811), ('snapshots\\resnet50_csv_45.h5', 0.05, 0.42369674185463657), ('snapshots\\resnet50_csv_28.h5', 0.0, 0.41797258297258294), ('snapshots\\resnet50_csv_22.h5', 0.1, 0.40782312925170067), ('snapshots\\resnet50_csv_30.h5', 0.05, 0.40745030745030747), ('snapshots\\resnet50_csv_50.h5', 0.1, 0.4013157894736842), ('snapshots\\resnet50_csv_37.h5', 0.0, 0.39436633627810097), ('snapshots\\resnet50_csv_47.h5', 0.0, 0.3908092403628118), ('snapshots\\resnet50_csv_41.h5', 0.2, 0.38839285714285715), ('snapshots\\resnet50_csv_35.h5', 0.15000000000000002, 0.38822228496141536), ('snapshots\\resnet50_csv_36.h5', 0.1, 0.38399981614267326), ('snapshots\\resnet50_csv_43.h5', 0.05, 0.3828025149453721), ('snapshots\\resnet50_csv_17.h5', 0.15000000000000002, 0.3746598639455782), ('snapshots\\resnet50_csv_21.h5', 0.05, 0.37316799237981496), ('snapshots\\resnet50_csv_29.h5', 0.0, 0.3672226582940869), ('snapshots\\resnet50_csv_32.h5', 0.1, 0.3669642857142857), ('snapshots\\resnet50_csv_39.h5', 0.05, 0.3659983291562239), ('snapshots\\resnet50_csv_33.h5', 0.05, 0.36450650157546705), ('snapshots\\resnet50_csv_46.h5', 0.1, 0.3637418137418137), ('snapshots\\resnet50_csv_42.h5', 0.0, 0.3635427827546054), ('snapshots\\resnet50_csv_25.h5', 0.05, 0.36262793405650545), ('snapshots\\resnet50_csv_11.h5', 0.05, 0.3579434337837699), ('snapshots\\resnet50_csv_27.h5', 0.05, 0.3495562586818953), ('snapshots\\resnet50_csv_40.h5', 0.0, 0.3492804814233386), ('snapshots\\resnet50_csv_31.h5', 0.05, 0.348015873015873), ('snapshots\\resnet50_csv_38.h5', 0.0, 0.3360606404724052), ('snapshots\\resnet50_csv_18.h5', 0.05, 0.3308032303830623), ('snapshots\\resnet50_csv_16.h5', 0.1, 0.32845804988662136), ('snapshots\\resnet50_csv_14.h5', 0.05, 0.32814818234986304), ('snapshots\\resnet50_csv_26.h5', 0.1, 0.3254329004329004), ('snapshots\\resnet50_csv_19.h5', 0.05, 0.3204281712685074), ('snapshots\\resnet50_csv_15.h5', 0.0, 0.3152310924369747), ('snapshots\\resnet50_csv_20.h5', 0.1, 0.29930213464696226), ('snapshots\\resnet50_csv_10.h5', 0.05, 0.2901406742663109), ('snapshots\\resnet50_csv_13.h5', 0.1, 0.27293083900226756), ('snapshots\\resnet50_csv_24.h5', 0.1, 0.2708245722531437), ('snapshots\\resnet50_csv_12.h5', 0.1, 0.2673262853528508), ('snapshots\\resnet50_csv_23.h5', 0.1, 0.2638221955448846), ('snapshots\\resnet50_csv_04.h5', 0.25, 0.24969474969474967), ('snapshots\\resnet50_csv_09.h5', 0.05, 0.24739891704177416), ('snapshots\\resnet50_csv_05.h5', 0.2, 0.24424342105263158), ('snapshots\\resnet50_csv_06.h5', 0.15000000000000002, 0.23761446886446885), ('snapshots\\resnet50_csv_07.h5', 0.15000000000000002, 0.233078231292517), ('snapshots\\resnet50_csv_03.h5', 0.15000000000000002, 0.21793958962895502), ('snapshots\\resnet50_csv_01.h5', 0.05, 0.19410188317751345), ('snapshots\\resnet50_csv_02.h5', 0.05, 0.19065212731754366), ('snapshots\\resnet50_csv_08.h5', 0.15000000000000002, 0.18758503401360543)]
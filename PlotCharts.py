import matplotlib.pyplot as plt
import sys
import numpy as np
import plotly.plotly as py
import random

rgb25 = ['#000000',
         '#000022',
         '#000044',
         '#000066',
         '#000088',
         '#0000AA',
         '#0000CC',
         '#0000EE',
         '#660000',
         '#660022',
         '#660044',
         '#660066',
         '#660088',
         '#6600AA',
         '#6600CC',
         '#AA0000',
         '#AA0022',
         '#AA0055',
         '#AA00AA',
         '#EEAA00',
         '#EEAA22',
         '#EEAA44',
         '#EEAA66',
         '#EEAA88',
         '#EEAAAA'
         ]

rgb15 = ['0x002200',
         '0x002222',
         '0x002244',
         '0x002266',
         '0x002288',
         '0x0044AA',
         '0x0044CC',
         '0x0044EE',
         '0x660000',
         '0x663322',
         '0x663344',
         '0x663366',
         '0x660088',
         '0x6622AA',
         '0x6688CC'
         ]


def get_spaced_colors(n):
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def colors(n):
  c = get_spaced_colors(n)
  ret = []
  for i in range(n):
    r,g,b = c[i]
    rc = str(hex(r)).replace('0x','').lower()
    gc = str(hex(g)).replace('0x','').lower()
    bc = str(hex(b)).replace('0x','').lower()
    if len(rc)==1:
        rc='0'+rc
    if len(gc)==1:
        gc='0'+gc
    if len(bc)==1:
        bc='0'+bc

    color =rc+gc+bc
    color = '#'+color
    ret.append(color)
  return ret

def plot_bardiag(data, ds_label, name):
    multiple_bars = plt.figure()
    ax = plt.subplot(111)
    N = len(data[0])
    x = np.arange(N)
    width = 1 / 20
    ax.bar(x - 0.05, data[0], width, color="black", label=ds_label[0])
    ax.bar(x, data[1], width, color="orange", label=ds_label[1])
    ax.bar(x + 0.05, data[2], width, color="g", label=ds_label[2])
    ax.bar(x + 0.1, data[3], width, color="b", label=ds_label[3])
    ax.legend()
    ax.set_xlabel("Class Id")
    ax.set_ylabel("Percentage of dataset")
    ax.set_title(ds_label[0].split(" ")[0] + " feature distribution")
    plt.savefig("./"+name)


def plot_bardiag_var(data, ds_label, name, xlab = "Class Id", ylab = "Percentage of dataset", title="de_top15 feature distribution", width=1/20):
    multiple_bars = plt.figure()
    ax = plt.subplot(111)
    N = len(data[0])
    x = np.arange(N)
    half = len(data)//2
    rgb = colors(25)
    for i in range(len(data)):
        ax.bar(x + i*width-half*width, data[i], width, color=rgb[i], label=ds_label[i])
    # ax.bar(x - 0.05, data[0], width, color="black", label=ds_label[0])
    # ax.bar(x, data[1], width, color="orange", label=ds_label[1])
    # ax.bar(x + 0.05, data[2], width, color="g", label=ds_label[2])
    # ax.bar(x + 0.1, data[3], width, color="b", label=ds_label[3])
    # fontP = FontProperties()
    # fontP.set_size('small')
    art = []
    lgd = ax.legend(loc=10, bbox_to_anchor=(0.5, -0.22), ncol=5)
    art.append(lgd)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    plt.savefig("./"+name, additional_artists=art,
    bbox_inches="tight")


if __name__ == '__main__':
    # plot kaggle dstl distrib
    # map = {0: 0.6158303379629634, 1: 0.03869027777777777, 2: 0.004051777777777777,
    #    3: 0.008816611111111112, 4: 0.029345759259259262, 5: 0.1080003055555556, 6: 0.1890181712962963,
    #    7: 0.00506625, 8: 0.001048527777777778, 9: 1.1768518518518519e-05,
    #    10: 0.00012021296296296294}
    #
    # names = sorted(list(map.keys()))
    # values = list(map.values())
    # barlist = plt.bar(names, values, color=[(0, 0, 0),    # 0 = OTHER
    #                                         (0.835, 0.514, 0.0275),  # 1 = BUILDING
    #                                         (0.835, 0.55, 0.1176),  # 2 = MIS_STRUCTURES
    #                                         (0.29804, 0, 0.6),  # 3 = ROAD
    #                                         (0.3137, 0, 0.6274),  # 4 = TRACK
    #                                         (0, 0.6, 0),  # 5 = TREES
    #                                         (0.0392, 0.3922, 0.0392),  # 6 = CROPS
    #                                         (0, 0, 0.85),  # 7 = WATERWAY
    #                                         (0, 0, 0.98),  # 8 = STANDING_WATER
    #                                         (1, 1, 0.4),  # 9 = VEHICLE_LARGE
    #                                         (0.95, 0.95, 0.4314)])
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
    # plt.title('Kaggle DSTL Challenge feature distribution')
    # plt.xlabel("Class Id")
    # plt.ylabel("Percentage of the labels")
    # plt.show()

    # plot eu_top25 feature distrib

    label_order = ['unlabelled', 'building', 'wood', 'water', 'road', 'residential']
    c_dict = {'Leipsic': {'water': 0.05705580605655757, 'unlabelled': 0.16406976731240266, 'road': 0.11598346136717878, 'building': 0.1258078100645736, 'residential': 0.34150532175462023, 'wood': 0.19557783344466725}, 'Stuttgart': {'water': 0.0014161211311511908, 'unlabelled': 0.14440433088399013, 'road': 0.1222865954130482, 'building': 0.15027680360721443, 'residential': 0.2742244767312404, 'wood': 0.3073916722333556}, 'Munic': {'water': 0.00030089067022934755, 'unlabelled': 0.10966011467379203, 'road': 0.10973195279447787, 'building': 0.1145949621465152, 'residential': 0.25208060008906713, 'wood': 0.41363147962591856}, 'Dresden': {'water': 0.002634168336673347, 'unlabelled': 0.13842447116455123, 'road': 0.09954177800044547, 'building': 0.08733408483633932, 'residential': 0.28787275662435985, 'wood': 0.3841927410376308}, 'Hanover': {'water': 0.010186767980405255, 'unlabelled': 0.14664124916499663, 'road': 0.10815534958806508, 'building': 0.14965002226675572, 'residential': 0.33334378757515, 'wood': 0.252022823424627}, 'Berlin': {'water': 0.022438604987753286, 'unlabelled': 0.13154974949899795, 'road': 0.10533592184368727, 'building': 0.15895236584279673, 'residential': 0.393404664885326, 'wood': 0.18831869294143855}, 'Frankfurt': {'water': 0.013453033845468712, 'unlabelled': 0.12232456023157418, 'road': 0.11413675684702743, 'building': 0.1543818136272545, 'residential': 0.2853934591405029, 'wood': 0.31031037630817165}, 'Dortmund': {'water': 0.0065052827877978186, 'unlabelled': 0.1737399465597863, 'road': 0.10019461701180142, 'building': 0.12257145958583832, 'residential': 0.3247836283678466, 'wood': 0.2722050656869295}, 'Dusseldorf': {'water': 0.04999829659318637, 'unlabelled': 0.17276359385437537, 'road': 0.11165811623246501, 'building': 0.16822607437096418, 'residential': 0.31709000222667577, 'wood': 0.1802639167223335}, 'Hamburg': {'water': 0.050306418392340256, 'unlabelled': 0.17645556112224442, 'road': 0.10174067022934749, 'building': 0.1509726564239589, 'residential': 0.382320418615008, 'wood': 0.13820427521710094}, 'Duisburg': {'water': 0.047585986733001656, 'unlabelled': 0.18495077588249229, 'road': 0.11631329661217711, 'building': 0.16808374200426435, 'residential': 0.3439588959962094, 'wood': 0.1391073027718551}, 'Essen': {'water': 0.02824707192162102, 'unlabelled': 0.14968667891338233, 'road': 0.12030683032732135, 'building': 0.14212011801380542, 'residential': 0.3731896849254066, 'wood': 0.18644961589846368}, 'Bremen': {'water': 0.02225327321309286, 'unlabelled': 0.1848189156089956, 'road': 0.12843124025829436, 'building': 0.18166749053662887, 'residential': 0.3877347417056337, 'wood': 0.09509433867735474}, 'Nuremberg': {'water': 0.004691739033622801, 'unlabelled': 0.15615587842351372, 'road': 0.126414885326208, 'building': 0.18438174682698738, 'residential': 0.32738044978846575, 'wood': 0.20097530060120244}, 'Cologne': {'water': 0.04324748385660209, 'unlabelled': 0.15040459251837013, 'road': 0.09438683478067242, 'building': 0.12818426297038524, 'residential': 0.27083736918281004, 'wood': 0.3129394566911603}}
    c_keys = sorted(c_dict.keys())
    label = c_keys
    data = [[c_dict[k][l] for l in label_order] for k in c_keys]
    print(len(data))
    plot_bardiag_var(data, label, 'de_top15.png', xlab="Class Id", ylab="Percentage of the labels")
    c_dict_eu = {'Kiev': {'wood': 0.4161556389327714, 'road': 0.10056768913463225, 'unlabelled': 0.17264828857293302, 'water': 0.07389142092329297, 'building': 0.09027585571353339, 'residential': 0.14646110672283663}, 'Milan': {'wood': 0.12936707752768842, 'road': 0.13332438370846725, 'unlabelled': 0.22411599231868531, 'water': 0.0036888531618435127, 'building': 0.23371682743837058, 'residential': 0.2757868658449448}, 'Budapest': {'wood': 0.28093019296951816, 'road': 0.11436933997050151, 'unlabelled': 0.14424401425762043, 'water': 0.0029628441494591934, 'building': 0.17746379056047204, 'residential': 0.28002981809242855}, 'Minsk': {'wood': 0.2557410818713449, 'road': 0.13584185903354884, 'unlabelled': 0.2145803939673745, 'water': 0.048429316712834715, 'building': 0.11708598030163127, 'residential': 0.22832136811326567}, 'Kazan': {'wood': 0.06422631637661758, 'road': 0.12659712182061578, 'unlabelled': 0.2159189089692103, 'water': 0.12553255243195002, 'building': 0.1531366912092816, 'residential': 0.314588409192325}, 'Kharkiv': {'wood': 0.4114804545454542, 'road': 0.08973225589225585, 'unlabelled': 0.20285044893378226, 'water': 0.02367941638608305, 'building': 0.11023688552188564, 'residential': 0.16202053872053876}, 'Prague': {'wood': 0.5202001262626266, 'road': 0.092898970959596, 'unlabelled': 0.22791936868686885, 'water': 0.016695650252525254, 'building': 0.08149719696969697, 'residential': 0.060788686868686846}, 'London': {'wood': 0.12402612286890063, 'road': 0.13444233392122287, 'unlabelled': 0.19045192239858932, 'water': 0.04674236037624927, 'building': 0.2072945208700764, 'residential': 0.297042739564962}, 'Barcelona': {'wood': 0.42419893939393943, 'road': 0.11519689393939392, 'unlabelled': 0.21064454545454545, 'water': 0.0015134722222222223, 'building': 0.21120968434343448, 'residential': 0.03723646464646464}, 'Hamburg': {'wood': 0.13746782259966983, 'road': 0.10125533734371304, 'unlabelled': 0.17755550247699922, 'water': 0.05202346661948575, 'building': 0.15073701344656754, 'residential': 0.3809608575135647}, 'Paris': {'wood': 0.1257192673992673, 'road': 0.13372442307692323, 'unlabelled': 0.06965909035409036, 'water': 8.654456654456655e-05, 'building': 0.2880231043956046, 'residential': 0.38278757020757065}, 'Bucharest': {'wood': 0.06173823345817728, 'road': 0.14394609862671653, 'unlabelled': 0.03907812109862671, 'water': 0.0, 'building': 0.23390874687890142, 'residential': 0.5213287999375783}, 'Nizhny Novgorod': {'wood': 0.07208014048531292, 'road': 0.14195257822477647, 'unlabelled': 0.24014583333333356, 'water': 0.04737652458492976, 'building': 0.17918338122605365, 'residential': 0.3192615421455937}, 'Madrid': {'wood': 0.23048596096096116, 'road': 0.17967824074074037, 'unlabelled': 0.16639200450450445, 'water': 0.0024961586586586576, 'building': 0.21029095970970957, 'residential': 0.2106566754254254}, 'St. Petersburg': {'wood': 0.1657028841245534, 'road': 0.16108467542964075, 'unlabelled': 0.2107680640632978, 'water': 0.0814950740173559, 'building': 0.1695483239748169, 'residential': 0.21140097839033514}, 'Vienna': {'wood': 0.17598641287527084, 'road': 0.12793948467966576, 'unlabelled': 0.17942674868461775, 'water': 0.04145593082636953, 'building': 0.21573835886722384, 'residential': 0.259453064066852}, 'Moscow': {'wood': 0.17619929039659143, 'road': 0.14859764503441475, 'unlabelled': 0.20719112913798782, 'water': 0.020240221238938053, 'building': 0.16513091281547015, 'residential': 0.2826408013765979}, 'Istanbul': {'wood': 0.3773206505071948, 'road': 0.11234363941967462, 'unlabelled': 0.27707076256192514, 'water': 0.04237619426751592, 'building': 0.16275271880160422, 'residential': 0.0281360344420854}, 'Belgrade': {'wood': 0.33256997892201134, 'road': 0.08512808641975309, 'unlabelled': 0.2275244354110206, 'water': 0.10735883769948812, 'building': 0.10460812255344777, 'residential': 0.14281053899427887}, 'Warsaw': {'wood': 0.24659035181236677, 'road': 0.14126459369817582, 'unlabelled': 0.1777814321250889, 'water': 0.004905751006870409, 'building': 0.14027014925373135, 'residential': 0.28918772210376686}, 'Munic': {'wood': 0.41681502267573706, 'road': 0.10923769274376419, 'unlabelled': 0.10840848072562358, 'water': 0.0003064172335600907, 'building': 0.1134388151927437, 'residential': 0.25179357142857156}, 'Samara': {'wood': 0.1369516987179487, 'road': 0.09573124999999993, 'unlabelled': 0.2095779487179488, 'water': 0.11764892094017095, 'building': 0.15502631410256407, 'residential': 0.28506386752136775}, 'Rome': {'wood': 0.2461802906818799, 'road': 0.12352319342569969, 'unlabelled': 0.18596800461831034, 'water': 0.0017813977180114093, 'building': 0.2031498234175498, 'residential': 0.2393972901385492}, 'Berlin': {'wood': 0.18662627032520332, 'road': 0.10508517389340549, 'unlabelled': 0.13250116869918696, 'water': 0.022604025519421866, 'building': 0.1587027100271003, 'residential': 0.3944806515356818}, 'Sofia': {'wood': 0.1615091104497354, 'road': 0.1412565228174603, 'unlabelled': 0.16760161210317456, 'water': 0.0007438244047619047, 'building': 0.18489391534391542, 'residential': 0.3439950148809523}}
    c_keys_eu = sorted(c_dict_eu.keys())
    label_eu = c_keys_eu
    data_eu = [[c_dict_eu[k][l] for l in label_order] for k in c_keys_eu]
    print(len(data_eu))
    plot_bardiag_var(data_eu, label_eu, 'eu_top25.png',xlab="", ylab="Percentage of the labels", width=1/35,
                     title="eu_top25 feature distribution")

    # label1 = "de_top14 complete", "de_top14 train", "de_top14 test", "de_top14 val"
    # # unlabelled building wood water road residential
    # data1 = [[0.1515, 0.1441, 0.2457, 0.0224, 0.1113, 0.3250],
    #         [0.1529, 0.1429, 0.2480, 0.0240, 0.1105, 0.3218],
    #         [0.1476, 0.1456, 0.2563, 0.0157, 0.1122, 0.3225],
    #         [0.1485, 0.1478, 0.2360, 0.0205, 0.1133, 0.3340]]
    # name1 = "de_top14_feature_distrib.png"
    # plot_bardiag(data1, label1, name1)
    #
    # label2 = "eu_top25_exde complete", "eu_top25_exde train", "eu_top25_exde test", "eu_top25_exde val"
    # # unlabelled building wood water road residential
    # data2 = [[0.1919, 0.1759, 0.2231, 0.0333, 0.1287, 0.2471],
    #         [0.1928, 0.1794, 0.2144, 0.0323, 0.1307, 0.2504],
    #         [0.1954, 0.1795, 0.1976, 0.0360, 0.1320, 0.2594],
    #         [0.1931, 0.1756, 0.2222, 0.0300, 0.1315, 0.2475]]
    # name2 = "eu_top25_exde_feature_distrib.png"
    # plot_bardiag(data2, label2, name2)
    #
    # label3 = "world_tiny2k complete", "world_tiny2k train", "world_tiny2k test", "world_tiny2k val"
    # # unlabelled building wood water road residential
    # data3 = [[0.1524, 0.0924, 0.1938, 0.0306, 0.1157, 0.4151],
    #         [0.1517, 0.0923, 0.1941, 0.0256, 0.1163, 0.4194],
    #         [0.1617, 0.0860, 0.2260, 0.0447, 0.1087, 0.3729],
    #         [0.1452, 0.0997, 0.1786, 0.0393, 0.1146, 0.4426]]
    # name3 = "world_tiny2k_feature_distrib.png"
    # plot_bardiag(data3, label3, name3)


## de_top14: complete

# {'water': 0.0224, 'wood': 0.2457, 'unlabelled': 0.1515, 'residential': 0.3250, 'road': 0.1113, 'building': 0.1441}
# de_top14: train
# {'water': 0.0240, 'wood': 0.2480, 'unlabelled': 0.1529, 'residential': 0.3218, 'road': 0.1105, 'building': 0.1429}
# de_top14: test
# {'water': 0.0157, 'wood': 0.2563, 'unlabelled': 0.1476, 'residential': 0.3225, 'road': 0.1122, 'building': 0.1456}
# de_top14: val
# {'water': 0.0205, 'wood': 0.2360, 'unlabelled': 0.1485, 'residential': 0.3340, 'road': 0.1133, 'building': 0.1478}
# _______________
#
#
# ## eu_top25_exde: complete
#
# {'water': 0.0333, 'wood': 0.2231, 'unlabelled': 0.1919, 'residential': 0.2471, 'road': 0.1287, 'building': 0.1759}
# eu_top25_exde: train
# {'water': 0.0323, 'wood': 0.2144, 'unlabelled': 0.1928, 'residential': 0.2504, 'road': 0.1307, 'building': 0.1794}
# eu_top25_exde: test
# {'water': 0.0360, 'wood': 0.1976, 'unlabelled': 0.1954, 'residential': 0.2594, 'road': 0.1320, 'building': 0.1795}
# eu_top25_exde: val
# {'water': 0.0300, 'wood': 0.2222, 'unlabelled': 0.1931, 'residential': 0.2475, 'road': 0.1315, 'building': 0.1756}
#
#
# ## world_tiny2k: complete
#
# {'water': 0.0306, 'wood': 0.1938, 'unlabelled': 0.1524, 'residential': 0.4151, 'road': 0.1157, 'building': 0.0924}
# world_tiny2k: train
# {'water': 0.0256, 'wood': 0.1941, 'unlabelled': 0.1517, 'residential': 0.4194, 'road': 0.1163, 'building': 0.0923}
# world_tiny2k: test
# {'water': 0.0447, 'wood': 0.2260, 'unlabelled': 0.1617, 'residential': 0.3729, 'road': 0.1087, 'building': 0.0860}
# world_tiny2k: val
# {'water': 0.0393, 'wood': 0.1786, 'unlabelled': 0.1452, 'residential': 0.4226, 'road': 0.1146, 'building': 0.0997}

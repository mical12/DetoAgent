import sys
sys.path.append('active_subspaces/active_subspaces')
sys.path.append('active_subspaces/active_subspaces/utils')
#import active_subspaces.active_subspaces as ac
from active_subspaces import subspaces as ac
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np     
from utils.response_surfaces import PolynomialApproximation
import sys
import scipy.optimize as opt

from scipy.optimize import minimize
from scipy.interpolate import griddata, Rbf
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import math
import os
from scipy.optimize import curve_fit

## 参数影响因子分析
# 可行域的上下界

# lower_bound = [30]
# upper_bound = [60]

# # 示例数据
# dependent_values = [28.52935028076172, 22.606800079345703, 30.669803619384766, 24.2705020904541, 26.463947296142578, 27, 26, 28, 29, 23, 24]
# samples = [[10.862114493461116, 1, 0.5], [8.381845766373827, 2, 0.3], [11.75423727553065,3, 0.7], [9.081219005822863,4, 0.9], [10.0,5, 0.1],[10.5,4.5, 0.35],[9.5,3.4,0.55],[11,2.6,0.75],[11.5,1.4,0.25], [8.5, 3.5, 0.2],[9.0, 5.6, 0.85] ]
# dependent_values_real = [28.52935028076172, 22.606800079345703, 30.669803619384766, 24.2705020904541, 26.463947296142578]
# ## 代理模型预测
# samples_1d = [[10.862114493461116], [8.381845766373827], [11.75423727553065], [9.081219005822863],[10.0]]
# dependent_values_real= [293.7179870605469, 1264.9298095703125, 293.5950012207031, 1226.7979736328125, 1004.0355224609375]
# samples_1d = [[56.81064527258827], [37.2130997518331], [51.71412213771896], [39.27716319268728], [45.0]]

# dependent_values = [0.05388620123267174, 0.04958821088075638, 0.05090159922838211, 0.053250525146722794, 0.05288241431117058, 0.030810575932264328, 0.0563897006213665, 0.03134777396917343, 0.03936201333999634, 0.05148660019040108, 0.031083988025784492, 0.04789670184254646, 0.04991242662072182, 0.04744672402739525, 0.05106889829039574, 0.03476977348327637, 0.047400373965501785, 0.050384048372507095, 0.043882325291633606, 0.04930239915847778, 0.035704463720321655, 0.03774385154247284, 0.03707071393728256, 0.051278237253427505, 0.05430733785033226, 0.03239298611879349, 0.0528305247426033, 0.04631885141134262, 0.05249734967947006, 0.05591251328587532, 0.03218703716993332, 0.045449163764715195, 0.04807113856077194]
# samples = [[26.978389475728918, 0.0004253511216446367, 2.2253805115544526e-06, 0.631807254549866], [21.10649083377037, 0.0006819399680573642, 7.471333605060974e-06, 0.6664271529092382], [22.134585028662435, 0.00017889664955830103, 5.1634107405695144e-06, 0.8789606153516107], [26.337456782880842, 0.0005218787076683967, 7.830058097147078e-06, 0.6769725605062487], [25.753351331351027, 0.0005718622098510971, 6.540879677985158e-06, 0.7093931458190965], [10.406048376453477, 0.0007436053998518051, 4.911075034524913e-06, 0.8064758486375169], [28.88252689812855, 0.0001874774675202962, 3.650636367655333e-06, 0.9959218012930077], [11.168185544801531, 0.0007607403660823355, 5.5983546382589435e-06, 0.9395125139279944], [15.628283717051767, 0.00012875591939826331, 9.721271043563421e-06, 0.8982419948793161], [23.801653215909646, 0.00040647167501679083, 6.772219617377285e-06, 0.9192101953654572], [11.277168498141595, 0.0009261185567871278, 6.121437518747774e-06, 0.9556114630283468], [19.887615651420713, 0.0001203098399017888, 1.6620956674787648e-06, 0.6561716721049153], [20.58844494746367, 0.0005786707897802928, 7.094325906869198e-06, 0.6247588676733294], [19.117959557183617, 0.000893769172215413, 2.8119577332984193e-06, 0.8433637104359488], [22.993266075040665, 0.0004484381304144014, 3.028479137495173e-06, 0.9794898318787015], [13.858418855842258, 0.0003044164115836248, 4.363705116262381e-06, 0.8727202719191542], [17.16323297277423, 0.0002916275439736805, 8.147254524886197e-06, 0.7901153239238156], [21.35151060321023, 0.0008864943891154842, 9.034117906748454e-06, 0.725144388973332], [16.77374385211556, 0.0009561076897530961, 8.400468200026867e-06, 0.7785428634535675], [29.87568714646964, 0.0007945816697687685, 2.6613335020175997e-06, 0.7708087234029762], [13.394463202061258, 0.0002548409134561576, 3.928185657744538e-06, 0.6446069755866644], [15.20889919953585, 0.000818073743991189, 9.205670545142013e-06, 0.8264038236024372], [14.893057887750965, 0.0005170418205690989, 3.3361557672389866e-06, 0.7456514786428436], [23.52655480853844, 0.0009746620015612479, 1.0934854578383998e-06, 0.9088962937666021], [28.131585073457412, 0.0008391647053582279, 4.59682752895421e-06, 0.8521329798086563], [12.874130839178846, 0.00032516240308002296, 9.654768942604683e-06, 0.6922868258124965], [25.45643677818653, 0.0003788958799800422, 8.817206546291652e-06, 0.7565699310611572], [18.56202583019155, 0.000646013989559833, 1.3853715933496091e-06, 0.9691397079219031], [24.88856181839323, 0.0006327369686702093, 7.246680724156867e-06, 0.7214726619806443], [28.108109189320473, 0.000217717668458547, 2.0391400867417167e-06, 0.8185044216311985], [12.039556495029377, 0.0007158046476925204, 5.975553742694024e-06, 0.9298731775378335], [17.538068907680298, 0.00047187496795669814, 5.405279742014285e-06, 0.6036541393106547], [20.0, 0.00055, 5.500000000000001e-06, 0.8]]


# x_input = np.array([[10.862114493461116, 1, 0.3]])  # 输入为一个 1x2 的样本
# in_labels = [ "temperature_difference_between_hot_and_cold",
#         "k_of_all_boundarys",
#         "epsilon_of_all_boundarys",
#         "Prt_of_all_boundarys_in_alphat"]
# in_labels = [ "temperature",
#         "k",
#         "epsilon",
#         "Prt"]
# ## 优化/参数标定
# f_target = 28.5253

#pitzDaily_paras_yplus_1vars
# lower_bound = [8.0]
# upper_bound = [12.0]
# in_labels_x = ["inlet_flow_velocity"]
# in_labels_y = ["max_yplus"]
# dependent_values = [28.52935028076172, 22.606800079345703, 30.669803619384766, 24.2705020904541, 26.463947296142578]
# samples = [[10.862114493461116], [8.381845766373827], [11.75423727553065], [9.081219005822863], [10.0]]

# #pitzDaily_paras_yplus_1vars_opt
# target = 25
#pitzDaily_paras_2vars
# lower_bound = [8.0, 0.3]
# upper_bound = [12.0, 0.45]
# in_labels = [ "inlet_flow_velocity",
#         "inlet_turbulent_kinetic_energy"]
# dependent_values = [23.13285255432129, 24.601346969604492, 27.759653091430664, 26.35544776916504, 21.582347869873047, 30.38140296936035, 29.39609718322754, 26.275800704956055, 26.14664649963379]
# samples = [[8.760647782075349, 0.3666276616934038], [9.279565324243272, 0.4476454194104561], [10.647595335012252, 0.3949695690453749], [10.073999096702874, 0.38701478258304733], [8.136883726125559, 0.3504265103731084], [11.811039528744796, 0.30631273473361603], [11.381644562303517, 0.3336637940096426], [9.995580817826207, 0.43043459964471387], [10.0, 0.375]]

# #counterFlowFlame2D_paras_30-60
# lower_bound = [10]
# upper_bound = [60]
# in_labels_x = ["inlet_velocity"]
# in_labels_y = ["max_temperature"]
# dependent_values = [293.5769958496094, 293.7405090332031, 1270.80078125, 1430.469970703125, 1469.1324462890625, 1795.5274658203125]
# samples = [[50.91548883538665], [57.707178151168364], [36.83143284787384], [27.583539517008898], [25.523031085128764], [13.857623038114944]]

# target = 1000
# #counterFlowFlame2D_paras_2vars
# in_labels = ["inlet_velocity",
#         "inlet_temperature"]
# lower_bound = [
#         10.0,
#         243.0]
# upper_bound = [
#         60.0,
#         343.0]
# dependent_values = [1354.1875, 1197.550048828125, 305.76300048828125, 259.2439880371094, 1607.5, 279.64849853515625, 1786.137451171875, 1427.7900390625, 293.5920104980469]
# samples = [[32.17581715214046, 318.7189181850554], [42.829045239264524, 340.76086792321973], [52.22890062958777, 305.16387787145055], [38.4223530830492, 258.9050202116821], [19.827280481787398, 311.8573024237744], [58.76332677025988, 278.8692330385342], [14.911918896065185, 245.24642231912065], [27.712288392015793, 287.06980469760873], [35.0, 293.0]]


# hydrogen combustion chamber
# in_labels = ["equivalenceRatio",
#         "initial_turbulent_kinetic_energy",
#         "ignition_duration"]
# lower_bound = [
#         0.5,
#         1.0,
#         0.0]
# upper_bound = [
#         1.5,
#         10.0,
#         0.002]
# dependent_values = [0.07071067917232597, 0.07071067917232597, 0.011180339845848937, 0.022360679691697874, 0.07071067917232597, 0.05883026468658933, 0.039012819211878116, 0.01702938729099328, 0.07071067917232597, 0.07071067917232597, 0.05936328976548208, 0.07071067917232597, 0.0, 0.01941648728024484, 0.03330165181160302, 0.03132091877313183, 0.0, 0.03820994515379319, 0.07071067917232597, 0.07071067917232597, 0.03962322537881257, 0.07071067917232597, 0.027166156113362996, 0.0, 0.06280127372626905]
# samples = [[1.4436960088645052, 7.901605875528435, 0.0017465031015095555], [1.3802855463611556, 6.0953453659713075, 0.0005484928354301654], [0.566040466954282, 2.7753563293894663, 0.001152263109818842], [0.7321120668243414, 7.480874721751601, 0.0007716103583857366], [1.3159580753798923, 3.404610620335912, 0.0018921419434173024], [0.9479659602364838, 5.033815353306637, 0.0015338755545217261], [1.0473601553464786, 1.4355325492227093, 0.000165596097711869], [0.6519306504624789, 5.611997419855954, 0.0010775617501516937], [1.180346730252774, 4.334968050092526, 0.00020029408746530223], [1.131037622413356, 5.1921634988489815, 0.0004148493157180102], [1.0212714696513912, 9.002623485733926, 0.000943873628213754], [1.3537497179131188, 4.516738665548296, 0.001186788322795074], [1.2492873179269361, 8.534816716311425, 4.7583001170258465e-05], [0.6951629658351535, 6.645212053880418, 0.0014149590721706533], [0.7891158788734087, 6.5937048000860745, 0.0014778239033648181], [0.8247199780860837, 2.321236075536421, 0.0016590134933953588], [0.612841794145095, 9.463623016817463, 0.001304149324130434], [0.9703468432138087, 1.8284338649915366, 0.0002639480847406601], [1.090923485999213, 3.7478109469113807, 0.001792094669811254], [1.4872156046434943, 9.803569335572545, 0.0006044694442301898], [0.8566167800282136, 2.9759994209198206, 0.001990066807861042], [1.265774707350869, 8.421706828733765, 0.0006864947230630656], [0.8892214279115378, 1.103054886102156, 0.0004451159251137172], [0.5278576883558509, 7.2582546315684615, 0.000835458923307845], [1.0, 5.5, 0.001]]

#dependent_values = [1354.1875, 1197.550048828125, 305.76300048828125, 259.2439880371094, 1607.5, 279.64849853515625, 1786.137451171875, 1427.7900390625, 293.5920104980469]
#samples = [[32.17581715214046, 400], [42.829045239264524, 380], [52.22890062958777, 305.16387787145055], [38.4223530830492, 258.9050202116821], [19.827280481787398, 350.8573024237744], [58.76332677025988, 278.8692330385342], [14.911918896065185, 350.24642231912065], [27.712288392015793, 340.06980469760873], [35.0, 293.0]]

#boxTurb_paras_1_var
# lower_bound = [0.01]
# upper_bound = [0.1]
# in_labels_y = ["average_turbulent_kinetic_energy"]
# in_labels_x = ["laminar_viscosity_nu_in_physicalProperties"]
# dependent_values= [0.0011805576848011384, 0.02057430899670545, 0.0006955987326689336, 0.0019384355648275878, 0.001353517310686242, 0.0037141189210782894, 0.0009746284106072278, 0.005037536157729965, 0.001798168120211554]
# samples = [[0.07023162489708767], [0.011580636506630371], [0.0942865997688584], [0.05259986644542641], [0.0649359414674977], [0.035385158493060904], [0.0782630015806249], [0.029268272539796372], [0.055]]

# # #boxTurb_paras_1_var_opt
# target = 0.01

#buoyantCavity_paras_1vars
# lower_bound = [10]
# upper_bound = [30]
# dependent_values = [0.0826786607503891, 0.029967600479722023, 0.050999775528907776, 0.04889822378754616]
# samples = [[29.433052528145076], [10.055004231168752], [22.10821593746259], [19.549590715628923]]
# in_labels_x = [ "temperature_difference"]
# in_labels_y = [ "max_velocity_in_X_direction"]
# target = 0.07

# #buoyantCavity_paras_4vars
# save_path = '/data/Chenyx/MetaOpenFOAM3'
# lower_bound = [
#         10.0,
#         0.0001,
#         1e-06,
#         0.6]
# upper_bound = [
#         30.0,
#         0.001,
#         1e-05,
#         1.0]
# in_labels = [ "temperature",
#         "k",
#         "epsilon",
#         "Prt"]

# dependent_values = [0.05388620123267174, 0.04958821088075638, 0.05090159922838211, 0.053250525146722794, 0.05288241431117058, 0.030810575932264328, 0.0563897006213665, 0.03134777396917343, 0.03936201333999634, 0.05148660019040108, 0.031083988025784492, 0.04789670184254646, 0.04991242662072182, 0.04744672402739525, 0.05106889829039574, 0.03476977348327637, 0.047400373965501785, 0.050384048372507095, 0.043882325291633606, 0.04930239915847778, 0.035704463720321655, 0.03774385154247284, 0.03707071393728256, 0.051278237253427505, 0.05430733785033226, 0.03239298611879349, 0.0528305247426033, 0.04631885141134262, 0.05249734967947006, 0.05591251328587532, 0.03218703716993332, 0.045449163764715195, 0.04807113856077194]
# samples = [[26.978389475728918, 0.0004253511216446367, 2.2253805115544526e-06, 0.631807254549866], [21.10649083377037, 0.0006819399680573642, 7.471333605060974e-06, 0.6664271529092382], [22.134585028662435, 0.00017889664955830103, 5.1634107405695144e-06, 0.8789606153516107], [26.337456782880842, 0.0005218787076683967, 7.830058097147078e-06, 0.6769725605062487], [25.753351331351027, 0.0005718622098510971, 6.540879677985158e-06, 0.7093931458190965], [10.406048376453477, 0.0007436053998518051, 4.911075034524913e-06, 0.8064758486375169], [28.88252689812855, 0.0001874774675202962, 3.650636367655333e-06, 0.9959218012930077], [11.168185544801531, 0.0007607403660823355, 5.5983546382589435e-06, 0.9395125139279944], [15.628283717051767, 0.00012875591939826331, 9.721271043563421e-06, 0.8982419948793161], [23.801653215909646, 0.00040647167501679083, 6.772219617377285e-06, 0.9192101953654572], [11.277168498141595, 0.0009261185567871278, 6.121437518747774e-06, 0.9556114630283468], [19.887615651420713, 0.0001203098399017888, 1.6620956674787648e-06, 0.6561716721049153], [20.58844494746367, 0.0005786707897802928, 7.094325906869198e-06, 0.6247588676733294], [19.117959557183617, 0.000893769172215413, 2.8119577332984193e-06, 0.8433637104359488], [22.993266075040665, 0.0004484381304144014, 3.028479137495173e-06, 0.9794898318787015], [13.858418855842258, 0.0003044164115836248, 4.363705116262381e-06, 0.8727202719191542], [17.16323297277423, 0.0002916275439736805, 8.147254524886197e-06, 0.7901153239238156], [21.35151060321023, 0.0008864943891154842, 9.034117906748454e-06, 0.725144388973332], [16.77374385211556, 0.0009561076897530961, 8.400468200026867e-06, 0.7785428634535675], [29.87568714646964, 0.0007945816697687685, 2.6613335020175997e-06, 0.7708087234029762], [13.394463202061258, 0.0002548409134561576, 3.928185657744538e-06, 0.6446069755866644], [15.20889919953585, 0.000818073743991189, 9.205670545142013e-06, 0.8264038236024372], [14.893057887750965, 0.0005170418205690989, 3.3361557672389866e-06, 0.7456514786428436], [23.52655480853844, 0.0009746620015612479, 1.0934854578383998e-06, 0.9088962937666021], [28.131585073457412, 0.0008391647053582279, 4.59682752895421e-06, 0.8521329798086563], [12.874130839178846, 0.00032516240308002296, 9.654768942604683e-06, 0.6922868258124965], [25.45643677818653, 0.0003788958799800422, 8.817206546291652e-06, 0.7565699310611572], [18.56202583019155, 0.000646013989559833, 1.3853715933496091e-06, 0.9691397079219031], [24.88856181839323, 0.0006327369686702093, 7.246680724156867e-06, 0.7214726619806443], [28.108109189320473, 0.000217717668458547, 2.0391400867417167e-06, 0.8185044216311985], [12.039556495029377, 0.0007158046476925204, 5.975553742694024e-06, 0.9298731775378335], [17.538068907680298, 0.00047187496795669814, 5.405279742014285e-06, 0.6036541393106547], [20.0, 0.00055, 5.500000000000001e-06, 0.8]]


# 1 independent vars
# 如果只有1个indenpendent var 就敏感性分析只画图，优化就

def parameter_influence_optimazition_1d(lower_bound, upper_bound, samples, dependent_values, f_target,save_path, in_labels_x, in_labels_y):
    def interpolate_1d(samples, dependent_values, x):
        """
        一维样条插值函数。
        参数:
        - samples: 样本点 (1D array)
        - dependent_values: 样本点对应的目标值 (1D array)
        - x: 插值点 (1D array or scalar)
        
        返回:
        - 插值值
        """
        # 确保样本点和目标值是 1D 数组
        samples = np.asarray(samples).flatten()
        dependent_values = np.asarray(dependent_values).flatten()
        sorted_indices = np.argsort(samples)
        samples = samples[sorted_indices]
        dependent_values = dependent_values[sorted_indices]
        # 创建样条插值函数
        spline = CubicSpline(samples, dependent_values)
        return spline(x)
    
    def objective_1d(x, samples, dependent_values, f_target):
        """
        一维优化目标函数。
        """
        interpolated_value = interpolate_1d(samples, dependent_values, x)
        return (interpolated_value - f_target) ** 2
    def create_bounds(lower_bound, upper_bound):
        if len(lower_bound) == 1:  # 如果 lower_bound 只有一个元素
            bounds = [(lower_bound[0], upper_bound[0])]
        else:
            bounds = [(lower_bound[i], upper_bound[i]) for i in range(len(lower_bound))]
        return bounds

    # 绘图函数
    def plot_response_surface(samples, dependent_values, lower_bound, upper_bound, save_path):
        """
        绘制响应曲面。
        
        参数:
        - samples: 样本点 (1D array)
        - dependent_values: 样本点对应的目标值 (1D array)
        - lower_bound: 横坐标的最小值
        - upper_bound: 横坐标的最大值
        - save_path: 保存路径
        """
        # 创建插值函数
        samples = np.asarray(samples).flatten()
        dependent_values = np.asarray(dependent_values).flatten()

        # 确保 samples 是递增的
        sorted_indices = np.argsort(samples)
        samples = samples[sorted_indices]
        dependent_values = dependent_values[sorted_indices]
        # samples = np.asarray(samples).flatten()
        # dependent_values = np.asarray(dependent_values).flatten()
        # print(samples)
        # print(samples.shape)
        spline = CubicSpline(samples, dependent_values)
        
        # 创建横坐标范围
        x_vals = np.linspace(lower_bound, upper_bound, 500)
        y_vals = spline(x_vals)
        
        # 创建图像
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label="Response Surface", color='blue')
        plt.scatter(samples, dependent_values, color='red', label='Sample Points', zorder=5)
        plt.xlabel(f"{in_labels_x[0]}", fontsize=16, fontname='Times New Roman')
        plt.ylabel(f"{in_labels_y[0]}", fontsize=16, fontname='Times New Roman')
        plt.title("Response Surface", fontsize=16, fontname='Times New Roman')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True)
        
        # 保存图像
        figs_folder = os.path.join(save_path, "figs")
        os.makedirs(figs_folder, exist_ok=True)
        save_file = os.path.join(figs_folder, "response_surface.png")
        plt.savefig(save_file, dpi=300)
        plt.close()
        print(f"Figure saved to {save_file}")

    #plot_response_surface(samples, dependent_values, lower_bound, upper_bound, save_path=save_path)
    parameter_influence_1d(lower_bound, upper_bound, samples, dependent_values,save_path, in_labels_x, in_labels_y)
    bounds = create_bounds(lower_bound, upper_bound)
    result = minimize_scalar(objective_1d, bounds=bounds[0], args=(samples, dependent_values,f_target), method='bounded')
    # 输出结果
    print("优化后的样本点:", result.x)
    print("最小化的目标函数值:", result.fun)

    return result.x

def parameter_influence_1d(lower_bound, upper_bound, samples, dependent_values,save_path, in_labels_x, in_labels_y):
        # 绘图函数
    def plot_response_surface(samples, dependent_values, lower_bound, upper_bound, save_path):
        """
        绘制响应曲面。
        
        参数:
        - samples: 样本点 (1D array)
        - dependent_values: 样本点对应的目标值 (1D array)
        - lower_bound: 横坐标的最小值
        - upper_bound: 横坐标的最大值
        - save_path: 保存路径
        """
        # 创建插值函数
        samples = np.asarray(samples).flatten()
        dependent_values = np.asarray(dependent_values).flatten()

        # 确保 samples 是递增的
        sorted_indices = np.argsort(samples)
        samples = samples[sorted_indices]
        dependent_values = dependent_values[sorted_indices]
        # samples = np.asarray(samples).flatten()
        # dependent_values = np.asarray(dependent_values).flatten()
        # print(samples)
        # print(samples.shape)
        spline = CubicSpline(samples, dependent_values)
        
        # 创建横坐标范围
        x_vals = np.linspace(lower_bound, upper_bound, 500)
        y_vals = spline(x_vals)
        
        # 创建图像
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, y_vals, label="Response Surface", color='blue')
        plt.scatter(samples, dependent_values, color='red', label='Sample Points', zorder=5)
        plt.xlabel(f"{in_labels_x[0]}", fontsize=16, fontname='Times New Roman')
        plt.ylabel(f"{in_labels_y[0]}", fontsize=16, fontname='Times New Roman')
        plt.title("Response Surface", fontsize=16, fontname='Times New Roman')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True)
        
        # 保存图像
        figs_folder = os.path.join(save_path, "figs")
        os.makedirs(figs_folder, exist_ok=True)
        save_file = os.path.join(figs_folder, "response_surface_CubicSpline.png")
        plt.savefig(save_file, dpi=300)
        plt.close()
        print(f"Figure saved to {save_file}")

    def plot_response_surface_2(samples, dependent_values, lower_bound, upper_bound, save_path):
        """
        绘制响应曲面，仅连接样本点。

        参数:
        - samples: 样本点 (1D array)
        - dependent_values: 样本点对应的目标值 (1D array)
        - lower_bound: 横坐标的最小值
        - upper_bound: 横坐标的最大值
        - save_path: 保存路径
        """
        # 确保 samples 和 dependent_values 是 NumPy 数组并排序
        samples = np.asarray(samples).flatten()
        dependent_values = np.asarray(dependent_values).flatten()

        # 确保 samples 是递增的
        sorted_indices = np.argsort(samples)
        samples = samples[sorted_indices]
        dependent_values = dependent_values[sorted_indices]

        # 创建图像
        plt.figure(figsize=(8, 6))

        # 绘制样本点的连线
        plt.plot(samples, dependent_values, label="Sample Line", color='blue', marker='o')

        # 设置图形属性
        plt.xlabel(f"{in_labels_x[0]}", fontsize=16, fontname='Times New Roman')
        plt.ylabel(f"{in_labels_y[0]}", fontsize=16, fontname='Times New Roman')
        plt.title("Response Surface (Sample Line)", fontsize=16, fontname='Times New Roman')
        plt.legend(fontsize=12, loc='best')
        plt.grid(True)

        # 保存图像
        figs_folder = os.path.join(save_path, "figs")
        os.makedirs(figs_folder, exist_ok=True)
        save_file = os.path.join(figs_folder, "response_surface_sample_line.png")
        plt.savefig(save_file, dpi=300)
        plt.close()
        print(f"Figure saved to {save_file}")

    # 计算RMS误差
    def rms_error(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))

    # 线性模型
    def linear_model(x, a, b):
        return a * x + b

    # 多项式模型（2次多项式示例）
    def polynomial_model(x, a, b, c):
        return a * x**2 + b * x + c

    # 指数模型
    def exponential_model(x, a, b):
        return a * np.exp(b * x)

    # 对数模型
    def logarithmic_model(x, a, b):
        return a * np.log(x) + b

    # 幂次模型
    def power_model(x, a, b):
        return a * x**b

    # 定义Logistic函数
    def logarithmic_model2(x, L, k, x_0):
        return L / (1 + np.exp(-k * (x - x_0)))

    # 拟合并计算RMS
    def fit_models(samples, dependent_values):
        samples = np.asarray(samples).flatten()
        dependent_values = np.asarray(dependent_values).flatten()

        # 确保 samples 是递增的
        sorted_indices = np.argsort(samples)
        samples = samples[sorted_indices]
        dependent_values = dependent_values[sorted_indices]

        models = {
            "linear": linear_model,
            "polynomial": polynomial_model,
            "power": power_model,
            "logarithmic": logarithmic_model,
            "exponential": exponential_model,
            "logarithmic2": logarithmic_model2
        }

        rms_results = {}
        
        for model_name, model in models.items():
            try:
                # 使用curve_fit拟合模型
                if model_name == "logarithmic2":
                    initial_guess = [max(dependent_values), 1, np.median(samples)]
                    popt, _ = curve_fit(model, samples, dependent_values)
                    # 计算模型的预测值
                    fitted_values = model(samples, *popt)
                    # 计算RMS误差
                    rms = rms_error(dependent_values, fitted_values)
                    rms_results[model_name] = rms
                else:
                    popt, _ = curve_fit(model, samples, dependent_values)
                    # 计算模型的预测值
                    fitted_values = model(samples, *popt)
                    # 计算RMS误差
                    rms = rms_error(dependent_values, fitted_values)
                    rms_results[model_name] = rms
            except Exception as e:
                print(f"Error fitting {model_name}: {e}")
                rms_results[model_name] = np.inf  # 如果拟合失败，RMS设为无穷大
        
            # 根据优先级选择最合适的模型
        model_priority = ["linear", "polynomial", "power", "logarithmic", "exponential","logarithmic2"]
        selected_model = None
        
        for model_name in model_priority:
            rms = rms_results.get(model_name)
            if rms is not None and rms < 0.1:
                # 如果RMS小于0.1，优先选择该模型
                selected_model = model_name
                break
            elif selected_model is None or rms < rms_results[selected_model]:
                # 否则选择RMS最小的模型
                selected_model = model_name
        # 返回拟合误差最小的模型
        #best_model = min(rms_results, key=rms_results.get)
        return selected_model, rms_results
    
    def plot_best_fit(samples, dependent_values, selected_model, save_path, lower_bound, upper_bound):
        samples = np.asarray(samples).flatten()
        dependent_values = np.asarray(dependent_values).flatten()

        # 确保 samples 是递增的
        sorted_indices = np.argsort(samples)
        samples = samples[sorted_indices]
        dependent_values = dependent_values[sorted_indices]
        # 根据选择的模型绘制拟合结果
        models = {
            "linear": linear_model,
            "polynomial": polynomial_model,
            "power": power_model,
            "logarithmic": logarithmic_model,
            "exponential": exponential_model,
            "logarithmic2": logarithmic_model2
        }

        # 获取最佳模型函数
        best_model_func = models[selected_model]

        # 拟合最佳模型
        popt, _ = curve_fit(best_model_func, samples, dependent_values)
        x_vals = np.linspace(lower_bound, upper_bound, 500)  # 横坐标从 lower_bound 到 upper_bound
        fitted_values = best_model_func(x_vals, *popt)

        # 创建图形
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, fitted_values, label=f'{selected_model} Fit', color='blue')

        # 绘制实际样本点
        plt.scatter(samples, dependent_values, color='red', label='Sample Points')

        # 设置标签和标题
        plt.xlabel(f"{in_labels_x[0]}", fontsize=16, fontname="Times New Roman")
        plt.ylabel(f"{in_labels_y[0]}", fontsize=16, fontname="Times New Roman")
        plt.title(f"Best Fit: {selected_model} Model", fontsize=16, fontname="Times New Roman")
        plt.grid(True)
        # 设置图例

        plt.legend(fontsize=12, loc='best')
        # 确保figs文件夹存在
        os.makedirs(os.path.join(save_path, 'figs'), exist_ok=True)

        # 保存图形
        save_file = os.path.join(save_path, 'figs', 'response_surface.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Response surface plot saved at {save_file}")

    best_model, rms_results = fit_models(samples, dependent_values)
    # 输出最佳模型和RMS误差
    print(f"Best model: {best_model}")
    print("RMS Errors for each model:", rms_results)
    plot_best_fit(samples, dependent_values, best_model, save_path, lower_bound, upper_bound)
    plot_response_surface(samples, dependent_values, lower_bound, upper_bound, save_path=save_path)
    plot_response_surface_2(samples, dependent_values, lower_bound, upper_bound, save_path)

    return 0

# >= 2 independent vars
def is_font_available(font_name):
    available_fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
    return font_name in available_fonts

def plot_opts(savefigs=True, save_path='run', figtype='.png'):
    """A few options for the plots.

    Parameters
    ----------
    savefigs : bool
        save figures into a separate figs director
    figtype : str 
        a file extention for the type of image to save
        
    Returns
    -------
    opts : dict 
        the chosen options. The keys in the dictionary are `figtype`, 
        `savefigs`, and `font`. The `font` is a dictionary that sets the font 
        properties of the figures.
    """

    # make figs directory
    if savefigs:
        if not os.path.isdir(f'{save_path}/figs'):
            os.mkdir(f'{save_path}/figs')

    # set plot fonts
    font_name = 'Times New Roman' if is_font_available('Times New Roman') else 'DejaVu Sans'

    myfont = {'family' : font_name,
            'weight' : 'normal',
            'size' : 14}

    opts = {'figtype' : figtype,
            'savefigs' : savefigs,
            'myfont' : myfont,
            'save_path': save_path}

    return opts


def parameter_influence_optimazition(lower_bound, upper_bound, samples, dependent_values, f_target,save_path):

    def convert_to_and_XX(samples, ub, lb):

        # 将 samples 转换为 XX
        samples = np.array(samples)
        
        # 确保 samples 的形状是 (M, N)
        M = samples.shape[0]
        
        # Normalize the samples to the interval [-1, 1]
        XX = 2. * (samples - lb) / (ub - lb) - 1.0
        
        return XX

    lb = np.array(lower_bound)
    ub = np.array(upper_bound)
    x0 = (lb+ub)/2.0

    n = len(ub)    # n表示多少个independent_var
    M = len(samples) # M表示多少次采样
    nbot = math.ceil(M / 10)

    initial_guess_x = x0 # 初始猜测值

    f = np.array(dependent_values).reshape(-1, 1)

    f = (f-f_target)**2

    XX = convert_to_and_XX(samples, ub, lb)

    #Instantiate a subspace object

    ss = ac.subspaces.Subspaces()

    #Compute the subspace with a global linear model (sstype='OLS') and 100 bootstrap replicates
    ss.compute(X=XX, f=f, nboot=nbot, sstype='QPHD')

    eigenvecs = ss.eigenvecs[0].reshape(n, 1)

    opts = plot_opts(save_path = save_path)

    ac.utils.plotters.eigenvectors(ss.eigenvecs[0].reshape(n, 1), opts = opts)
    #This plots the eigenvalues (ss.eigenvals) with bootstrap ranges (ss.e_br)
    ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br, opts = opts)

    #This plots subspace errors with bootstrap ranges (all contained in ss.sub_br)
    ac.utils.plotters.subspace_errors(ss.sub_br, opts = opts)

    #This makes sufficient summary plots with the active variables (XX.dot(ss.W1)) and output (f)
    ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), f, opts = opts)

    #quadratic polynomial approximation
    RS = PolynomialApproximation(2)

    #Train the surface with active variable values (y = XX.dot(ss.W1)) and function values (f)
    y = XX.dot(ss.W1)
    RS.train(y, f)
    print ('The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr))

    #Plot the data and response surface prediction
    plt.figure(figsize=(7, 7))
    y0 = np.linspace(-2, 2, 200)

    plt.plot(y, f, 'bo', y0, RS.predict(y0[:,None])[0], 'k-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xlabel('Active Variable Value', fontsize=18)
    plt.ylabel('Output', fontsize=18)
    figname = f'{save_path}/figs/response_surface.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    # ---------------------- 优化参数 x 使得 y 最接近某个目标值 ----------------------

    def predict_with_proxy_model(x):
        """
        使用代理模型预测给定活性变量 x 的输出 y。
        """
        
        return RS.predict(x)[0]

    def objective_function(x, target_y):
        """
        目标函数：返回预测的 y 与目标值 target_y 的差距（平方误差）。
        目标是最小化这个误差，使得预测的 y 接近 target_y。
        """
        x = x.reshape(1,1)
        y_pred = predict_with_proxy_model(x)
        return np.sum((y_pred - target_y) ** 2)

    # 优化过程：使得 y 最接近目标值 target_y
    def optimize_x_for_target(target_y, initial_guess):
        """
        优化 x 使得代理模型输出 y 最接近给定的目标值 target_y
        """
        result = opt.minimize(objective_function, initial_guess, args=(target_y,), method='Nelder-Mead')
        optimized_x = result.x
        optimized_x = optimized_x.reshape(1,1)
        optimized_y = predict_with_proxy_model(optimized_x)
        print(f"优化后的 x: {optimized_x}, 优化后的 y: {optimized_y}")
        return optimized_x, optimized_y

    # 示例：希望优化 x，使得 y 最接近目标值 target_y
    target_y = 0 

    initial_guess = np.dot(initial_guess_x, eigenvecs)
    initial_guess = initial_guess.flatten()
    optimized_x, optimized_y = optimize_x_for_target(target_y, initial_guess)


    def interpolate(samples, dependent_values, points, method='cubic'):
        """
        通用插值函数，支持一维、二维以及多维输入。
        
        参数:
        - samples: 样本点，形状为 (M, n)，其中 M 是样本数量，n 是样本维度。
        - dependent_values: 样本点对应的目标值，形状为 (M,)。
        - points: 需要插值的点，形状为 (P, n)，其中 P 是插值点的数量。
        - method: 插值方法，可选 'linear', 'nearest', 'cubic'（默认值为 'cubic'）。
        
        返回:
        - 插值结果，形状为 (P,)。
        """
        # 确保 samples 是二维数组，形状为 (M, n)
        samples = np.atleast_2d(samples)
        if samples.shape[1] == 1:  # 如果是 (M, 1)，需要保持一致
            samples = samples.reshape(-1, 1)

        # 确保 dependent_values 是一维数组
        dependent_values = np.asarray(dependent_values).flatten()

        # 确保 points 是二维数组，形状为 (P, n)
        points = np.atleast_2d(points)
        if points.shape[1] == 1 and samples.shape[1] > 1:
            points = points.reshape(-1, 1)
        # 判断维度选择插值方法
            # 判断维度，选择插值方法
        if samples.shape[1] >= 3:  # 如果数据维度 >= 3，使用 RBF 插值
            #print(f"警告：使用 RBF 插值方法，适用于高维数据。")
            # 使用 RBF 插值
            rbf = Rbf(samples[:, 0], samples[:, 1], samples[:, 2], dependent_values, function='multiquadric', epsilon=2)
            return rbf(points[:, 0], points[:, 1], points[:, 2])
        else:
            # 对于二维或一维数据，使用 griddata
            if method == 'cubic' and samples.shape[1] == 2:
                # 对于二维数据，支持 cubic 插值
                return griddata(samples, dependent_values, points, method='cubic')
            else:
                # 对于其他情况使用 linear 或 nearest
                return griddata(samples, dependent_values, points, method='linear')

    # 定义目标函数
    def objective(x, samples, dependent_values):
        interpolated_values = interpolate(samples, dependent_values, x.reshape(1, -1))  # Reshape x to 2D array
        g = (interpolated_values - f_target) ** 2
        return g

    def constraint(x, eigenvecs, optimized_x, ub, lb):
        x = convert_to_and_XX(np.array([x]), ub, lb)
        y = np.dot(x, eigenvecs) - optimized_x

        return y[0] 

    def create_bounds(lower_bound, upper_bound):
        if len(lower_bound) == 1:  # 如果 lower_bound 只有一个元素
            bounds = [(lower_bound[0], upper_bound[0])]
        else:
            bounds = [(lower_bound[i], upper_bound[i]) for i in range(len(lower_bound))]
        return bounds

    # 约束条件：限制搜索空间在给定的上下界内

    bounds = create_bounds(lower_bound, upper_bound)

    # 优化函数的约束条件
    constraints = [{'type': 'eq', 'fun': constraint, 'args': (eigenvecs, optimized_x, ub, lb)}]
    samples_np = np.atleast_2d(samples)
    # 使用scipy.optimize.minimize来最小化目标函数
    #result = minimize(objective, x0, args=(samples, dependent_values), bounds=bounds, constraints=constraints, method='Nelder-Mead')
    if samples_np.shape[1] > 1:
        result = minimize(objective, x0, args=(samples, dependent_values), bounds=bounds, constraints=constraints, method='SLSQP')
    elif samples_np.shape[1] == 1:
        result = minimize(objective, x0, args=(samples, dependent_values), bounds=bounds, method='Nelder-Mead')

    # 输出优化结果
    print("优化后的样本点:", result.x)
    print("最小化的目标函数值:", result.fun)
    return [result.x]


def parameter_influence(lower_bound, upper_bound, samples, dependent_values, save_path, in_labels):

    def convert_to_and_XX(samples, ub, lb):

        # 将 samples 转换为 XX
        samples = np.array(samples)
        
        # 确保 samples 的形状是 (M, N)
        M = samples.shape[0]
        
        # Normalize the samples to the interval [-1, 1]
        XX = 2. * (samples - lb) / (ub - lb) - 1.0
        
        return XX

    lb = np.array(lower_bound)
    ub = np.array(upper_bound)
    x0 = (lb+ub)/2.0

    n = len(ub)    # 或者直接用 len 函数
    M = len(samples) # M表示多少次采样
    nbot = math.ceil(M / 10)

    initial_guess_x = (lb+ ub)/2 # 初始猜测值

    f = np.array(dependent_values).reshape(-1, 1)

    #f = (f-f_target)**2

    XX = convert_to_and_XX(samples, ub, lb)


    #Instantiate a subspace object

    ss = ac.subspaces.Subspaces()

    #Compute the subspace with a global linear model (sstype='OLS') and 100 bootstrap replicates
    ss.compute(X=XX, f=f, nboot=nbot, sstype='QPHD')

    eigenvecs = ss.eigenvecs[0].reshape(n, 1)
    print('eigenvecs:',eigenvecs)
    opts = plot_opts(save_path = save_path)

    ac.utils.plotters.eigenvectors(ss.eigenvecs[0].reshape(n, 1),in_labels = in_labels, opts = opts)
    #This plots the eigenvalues (ss.eigenvals) with bootstrap ranges (ss.e_br)
    ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br, opts = opts)

    #This plots subspace errors with bootstrap ranges (all contained in ss.sub_br)
    ac.utils.plotters.subspace_errors(ss.sub_br, opts = opts)

    #This makes sufficient summary plots with the active variables (XX.dot(ss.W1)) and output (f)
    ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), f, opts = opts)

    #quadratic polynomial approximation
    RS = PolynomialApproximation(2)

    #Train the surface with active variable values (y = XX.dot(ss.W1)) and function values (f)
    y = XX.dot(ss.W1)
    RS.train(y, f)
    print ('The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr))

    #Plot the data and response surface prediction
    plt.figure(figsize=(7, 7))
    y0 = np.linspace(-2, 2, 200)

    plt.plot(y, f, 'bo', y0, RS.predict(y0[:,None])[0], 'k-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xlabel('Active Variable Value', fontsize=18)
    plt.ylabel('Output', fontsize=18)
    figname = f'{save_path}/response_surface.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    return eigenvecs


def parameter_influence_predict(lower_bound, upper_bound, samples, dependent_values, x_input,save_path):

    def convert_to_and_XX(samples, ub, lb):

        # 将 samples 转换为 XX
        samples = np.array(samples)
        
        # 确保 samples 的形状是 (M, N)
        M = samples.shape[0]
        
        # Normalize the samples to the interval [-1, 1]
        XX = 2. * (samples - lb) / (ub - lb) - 1.0
        
        return XX

    lb = np.array(lower_bound)
    ub = np.array(upper_bound)
    x0 = (lb+ub)/2.0

    n = len(ub)    # 或者直接用 len 函数
    M = len(samples) # M表示多少次采样
    nbot = math.ceil(M / 10)

    initial_guess_x = (lb+ ub)/2 # 初始猜测值

    f = np.array(dependent_values).reshape(-1, 1)

    f = (f-f_target)**2

    XX = convert_to_and_XX(samples, ub, lb)

    #Instantiate a subspace object

    ss = ac.subspaces.Subspaces()

    #Compute the subspace with a global linear model (sstype='OLS') and 100 bootstrap replicates
    ss.compute(X=XX, f=f, nboot=nbot, sstype='QPHD')

    eigenvecs = ss.eigenvecs[0].reshape(n, 1)

    opts = plot_opts(save_path = save_path)

    ac.utils.plotters.eigenvectors(ss.eigenvecs[0].reshape(n, 1), opts = opts)
    #This plots the eigenvalues (ss.eigenvals) with bootstrap ranges (ss.e_br)
    ac.utils.plotters.eigenvalues(ss.eigenvals, ss.e_br, opts = opts)

    #This plots subspace errors with bootstrap ranges (all contained in ss.sub_br)
    ac.utils.plotters.subspace_errors(ss.sub_br, opts = opts)

    #This makes sufficient summary plots with the active variables (XX.dot(ss.W1)) and output (f)
    ac.utils.plotters.sufficient_summary(XX.dot(ss.W1), f, opts = opts)

    #quadratic polynomial approximation
    RS = PolynomialApproximation(2)

    #Train the surface with active variable values (y = XX.dot(ss.W1)) and function values (f)
    y = XX.dot(ss.W1)
    RS.train(y, f)
    print ('The R^2 value of the response surface is {:.4f}'.format(RS.Rsqr))

    #Plot the data and response surface prediction
    plt.figure(figsize=(7, 7))
    y0 = np.linspace(-2, 2, 200)

    plt.plot(y, f, 'bo', y0, RS.predict(y0[:,None])[0], 'k-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.xlabel('Active Variable Value', fontsize=18)
    plt.ylabel('Output', fontsize=18)
    figname = f'{save_path}/response_surface.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.0)

    
    def predict_with_proxy_model(x):
        """
        使用代理模型预测给定活性变量 x 的输出 y。
        """
        
        return RS.predict(x)[0]

    x_input_np = convert_to_and_XX(x_input, ub, lb)
    x_real = np.dot(x_input_np, eigenvecs)
    y_pred = predict_with_proxy_model(x_real)  # 预测结果

    print(f"parameter = {x_input}对应的响应面值 f = {y_pred}")
    return y_pred

# save_path = '/data/Chenyx/MetaOpenFOAM3/figs'
# optimate_x = parameter_influence_optimazition_1d(lower_bound, upper_bound, samples, dependent_values, target, save_path, in_labels_x, in_labels_y)
# print('optimate_x:',optimate_x)
#parameter_influence(lower_bound, upper_bound, samples, dependent_values, save_path, in_labels)
#optimate_x = parameter_influence_optimazition(lower_bound, upper_bound, samples, dependent_values, f_target, save_path)
#optimate_x = parameter_influence_optimazition_1d(lower_bound, upper_bound, samples_1d, dependent_values_real, f_target, save_path)
#print('optimate_x:',optimate_x)
#parameter_influence_1d(lower_bound, upper_bound, samples_1d, dependent_values_real,save_path)


# PROMPT_for_image_analyze: str = """
# This image illustrates the trend between {independent_var} and {dependent_var}. The user's requirement is {user_requirements}.
# Please analyze this image and respond to the user's requirement accordingly.
#     """
# independent_var = "temperature_difference"

# dependent_var = "max_velocity_in_X_direction"
# user_requirements = "Please help me analyze the effect of the temperature difference betwwen the hot and cold (from 10 K to 30 K) on the max velocity in X direction in a simulation: do a RANS simulation of buoyantCavity using buoyantFoam, which investigates natural convection in a heat cavity; the remaining patches are treated as adiabatic (except hot and cold patches)."

# # in_labels_x = ["inlet_flow_velocity"]
# # in_labels_y = ["max_yplus"]
# # in_labels_x = ["inlet_velocity"]
# # in_labels_y = ["max_temperature"]
# independent_var = "equivalenceRatio"

# dependent_var = "distance_from_origin_where_temperature_reaches_2000K"
# user_requirements = "Please help me analyze the effect of equivalenceRatio (from 0.5 to 1.5) on the distance from the origin, defined as d = sqrt{X^2 + Y^2}, where the temperature reaches 2000K at latest time through post-processing (if min(T) > 2000 K, then d = sqrt{max(X)^2 + max(Y)^2}; if max(T)<2000 K, then d = 0) in a simulation: Perform a 2D simulation of a hydrogen combustion chamber using a 50*50*1 grid with an end time of 0.005."

# prompt = PROMPT_for_image_analyze.format(independent_var = independent_var, dependent_var = dependent_var, user_requirements = user_requirements)
# print(prompt)


# PROMPT_for_image_analyze_multi_vars: str = """
# These two images analyze the relationship between the independent variables ({independent_vars}) and the dependent variable ({dependent_var}) using the Active Subspace Method. The first figure presents the response surface constructed by the Active Subspace Method, while the second figure illustrates the magnitude of influence of each parameter.
# The user's requirement is {user_requirements}.
# Please analyze two images and respond to the user's requirement accordingly.
#     """
# # independent_vars = [ "temperature_difference_between_hot_and_cold",
# #         "k_of_all_boundarys",
# #         "epsilon_of_all_boundarys",
# #         "Prt_of_all_boundarys_in_alphat"]
# # dependent_var = [ "max_velocity_in_X_direction"]

# independent_vars = [ "temperature_difference_between_hot_and_cold",
#         "k_of_all_boundarys",
#         "epsilon_of_all_boundarys",
#         "Prt_of_all_boundarys_in_alphat"]
# dependent_var = [ "max_velocity_in_X_direction"]
# user_requirements = "Please help me analyze the effect of the temperature_difference_betwwen_hot_and_cold (from 10 K to 30 K), k_of_all_boundarys (from 1e-04 to 1e-03), epsilon_of_all_boundarys (from 1e-06 to 1e-05) and Prt_of_all_boundarys_in_alphat (from 0.6 to 1.0) on the max velocity in X direction in a simulation: do a RANS simulation of buoyantCavity using buoyantFoam and kEpsilon turbulent model, which investigates natural convection in a heat cavity; the remaining patches are treated as adiabatic (except hot and cold patches)."
# independent_vars = ["equivalenceRatio",
#         "initial_turbulent_kinetic_energy",
#         "ignition_duration"]
# dependent_var = "distance_from_origin_where_temperature_reaches_2000K"
# # independent_vars = in_labels
# # dependent_var = "max_temperature"
# user_requirements = "Please help me analyze the effect of equivalenceRatio (from 0.5 to 1.5), inital_turbulent_kinetic_energy_in_0 (from 1 to 10), initial ignition duration time (from 0 to 0.002) on the distance from the origin, defined as d = sqrt{X^2 + Y^2}, where the temperature reaches 2000K at latest time through post-processing (if min(T) > 2000 K, then d = sqrt{max(X)^2 + max(Y)^2}; if max(T)<2000 K, then d = 0) in a simulation: Perform a 2D simulation of a hydrogen combustion chamber using a 50*50*1 grid with an end time of 0.005."
# prompt = PROMPT_for_image_analyze_multi_vars.format(independent_vars = independent_vars, dependent_var = dependent_var, user_requirements = user_requirements)
# print(prompt)
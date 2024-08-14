\Problem name: Ore Mixing Problem

Minimize
OBJROW: x_00 + x_10 + x_20 + x_30 + y_00 + y_01 + y_02 + y_03 + 232.55814 a_max_00 + 1000000000000 a_max_10
 + 100000000000 a_max_20 + 100000000000 a_max_30 + 100000000000 a_max_40 + 100000000000 a_max_50 + 10000000000 a_min_00 + 172413.79310 a_min_10 + 20408.16327 a_min_20 + 1428571.42857 a_min_30 + 10000 a_min_40 + 4000 a_min_50
 + 1000000 b_max_00 + 1000000 b_max_10 + 1000000 b_max_20 + 1000000 b_max_30 + 1000000 b_max_40 + 1000000 b_max_50 + 1000000 b_min_00 + 1000000 b_min_10 + 1000000 b_min_20 + 1000000 b_min_30
 + 1000000 b_min_40 + 1000000 b_min_50
Subject To
input_weight_constr_0:  y_00 + y_01 + y_02 + y_03 <= 90000
capacity_constr_0:  y_00 <= 80000
weight_constr_00:  x_00 - y_00 <= 120000
capacity_constr_1:  y_01 <= 10000
weight_constr_10:  x_10 - y_01 <= 90000
capacity_constr_2:  y_02 <= 100000
weight_constr_20:  x_20 - y_02 <= 100000
capacity_constr_3:  y_03 <= 10000
weight_constr_30:  x_30 - y_03 <= 30000
demand_constr_0:  x_00 + x_10 + x_20 + x_30 = 300000
min_quality_constr_00:  0.30000 x_00 -0.70000 x_10 + 0.30000 x_20 -0.70000 x_30 + 300000 a_min_00 >= -0
max_quality_constr_00:  -42.70000 x_00 -43.70000 x_10 -42.70000 x_20 -43.70000 x_30 -300000 a_max_00 <= -0
goal_quality_constr_00:  0.30000 x_00 -0.70000 x_10 + 0.30000 x_20 -0.70000 x_30 -300000 b_max_00 + 300000 b_min_00 = -0
min_quality_constr_10:  5.30000 x_00 + 5.60000 x_10 + 3.70000 x_20 + 3.80000 x_30 + 300000 a_min_10 >= -0
max_quality_constr_10:  -0.50000 x_00 -0.20000 x_10 -2.10000 x_20 -2 x_30 -300000 a_max_10 <= -0
goal_quality_constr_10:  -0.50000 x_00 -0.20000 x_10 -2.10000 x_20 -2 x_30 -300000 b_max_10 + 300000 b_min_10 = -0
min_quality_constr_20:  4.50000 x_00 + 4.80000 x_10 + 5 x_20 + 5.30000 x_30 + 300000 a_min_20 >= -0
max_quality_constr_20:  -0.40000 x_00 -0.10000 x_10 + 0.10000 x_20 + 0.40000 x_30 -300000 a_max_20 <= -0
goal_quality_constr_20:  -0.40000 x_00 -0.10000 x_10 + 0.10000 x_20 + 0.40000 x_30 -300000 b_max_20 + 300000 b_min_20 = -0
min_quality_constr_30:  0.05000 x_00 + 0.05000 x_10 + 0.05000 x_20 + 0.05000 x_30 + 300000 a_min_30 >= -0
max_quality_constr_30:  -0.02000 x_00 -0.02000 x_10 -0.02000 x_20 -0.02000 x_30 -300000 a_max_30 <= -0
goal_quality_constr_30:  -0.02000 x_00 -0.02000 x_10 -0.02000 x_20 -0.02000 x_30 -300000 b_max_30 + 300000 b_min_30 = -0
min_quality_constr_40:  8 x_00 + 8 x_10 + 2 x_20 + 2 x_30 + 300000 a_min_40 >= -0
max_quality_constr_40:  -2 x_00 -2 x_10 -8 x_20 -8 x_30 -300000 a_max_40 <= -0
goal_quality_constr_40:  -2 x_00 -2 x_10 -8 x_20 -8 x_30 -300000 b_max_40 + 300000 b_min_40 = -0
min_quality_constr_50:  20 x_00 + 20 x_10 + 15 x_20 + 15 x_30 + 300000 a_min_50 >= -0
max_quality_constr_50:  -5 x_00 -5 x_10 -10 x_20 -10 x_30 -300000 a_max_50 <= -0
goal_quality_constr_50:  -5 x_00 -5 x_10 -10 x_20 -10 x_30 -300000 b_max_50 + 300000 b_min_50 = -0
Bounds
End

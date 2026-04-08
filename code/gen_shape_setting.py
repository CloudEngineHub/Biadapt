import json
import numpy as np
# cs_file = open("../stats/shape_setting.json", "w")
cs_file = "../stats/shape_setting.json"
cate_field_name = [
    "obj_scale", "robot_scale", "urdf_type", 'state', "start_dist1", "displacement1", "maneuver_dist1", "start_dist2",
    "displacement2", "maneuver_dist2"
]

src_cate_setting = {
    "opening": {
        "common_params": {
            "damping": 10,
            "friction": 10,
            "density": 0.5,
            "mu": 3.0,
            "euler": [0, 0, 5 * np.pi / 6],
        },
        "shape_setting": {
            "10495": [0.8, 1.8, "origin", 'closed', 0.15, 0.04, 0, 0.15, 0.2, 0.0],  #Scissors
            "10502": [0.8, 1.8, "origin", 'closed', 0.15, 0.04, 0, 0.15, 0.2, 0.0],
            "10537": [0.8, 1.8, "origin", 'closed', 0.15, 0.04, 0, 0.15, 0.2, 0.0],
            "10558": [0.8, 1.8, "origin", 'closed', 0.15, 0.04, 0, 0.15, 0.2, 0.0],
            "10889": [0.8, 1.8, "origin", 'closed', 0.15, 0.04, 0, 0.15, 0.2, 0.0],
            "10907": [0.8, 1.8, "origin", 'closed', 0.15, 0.04, 0, 0.15, 0.2, 0.0],
        }
    },
    "unfolding": {
        "common_params": {
            "damping": 200,
            "friction": 100,
            "density": 1.0,
            "mu": 5.0,
            "euler": [0, 0, -1 / 4 * np.pi]
        },
        "shape_setting": {
            "9968": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],  #laptop
            "9992": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],
            "10040": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],
            "10211": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],
            "10213": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],
            "10243": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],
            "10269": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],
            "10885": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],
            "11248": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0],
            "11581": [0.8, 1.5, "origin", 'low-left', 0.1, 0.08, 0.04, 0.15, 0.4, 0]
        }
    },
    "uncapping": {
        "common_params": {
            "damping": 200,
            "friction": 10,
            "density": 0.5,
            "mu": 3.0,
            "euler": [0, 0, 0.5 * np.pi]
        },
        "shape_setting": {
            "100123": [0.8, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "100061": [0.8, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "100071": [0.6, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "100072": [0.6, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "100073": [0.8, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "100085": [0.8, 4.0, "origin", 'closed',0.35, 0.2, 0, 0.35, 0.2, 0],
            "100095": [0.6, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "100103": [0.8, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "100116": [0.6, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "100133": [0.8, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "102053": [0.8, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
            "102068": [0.8, 4.0, "origin", 'closed', 0.35, 0.2, 0, 0.35, 0.2, 0],
        }
    }
}
trg_cate_setting = {
    "opening": {
        "common_params": {
            "damping": 20,
            "friction": 20,
            "density": 1.0,
            "mu": 3.0,
            "euler": [0, 0, np.pi],
        },
        "shape_setting": {
            "100142": [0.8, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],  # Pliers
            "100144": [0.8, 1.8, "origin", 'closed', 0.18, 0.25, 0, 0.18, 0.25, 0.00],  # Pliers
            "100178": [1.2, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],
            "100188": [0.8, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],
            "102221": [0.8, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],
            "100182": [0.8, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],
            "102073": [0.7, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],
            "102242": [0.8, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],
            "102258": [1.0, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],
            "102260": [1.0, 1.8, "origin", 'closed', 0.15, 0.25, 0, 0.15, 0.25, 0.00],
        }
    },
    "unfolding": {
        "common_params": {
            "damping": 200,
            "friction": 100,
            "density": 0.8,
            "mu": 6.0,
            "euler": [0, 0, -np.pi / 4]
        },
        "shape_setting": {
            "100174": [0.6, 1.5, "processed", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0],
            "100189": [0.6, 1.5, "processed", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0],
            "100191": [0.8, 1.5, "processed", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0],
            "100685": [0.8, 1.5, "processed", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0],
            "102377": [0.6, 1.5, "processed", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0],
            "102379": [0.6, 1.5, "processed", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0],
            "100221": [0.8, 1.5, "origin", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0],
            "100141": [0.8, 1.2, "processed", 'low-left', 0.1, 0.03, 0.04, 0.10, 0.3, 0],
            "100214": [0.8, 1.2, "origin", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.4, 0],
            "100243": [0.8, 1.5, "processed", 'random-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0],
            "100671": [0.8, 1.5, "origin", 'low-left', 0.1, 0.06, 0.04, 0.13, 0.3, 0]
        }
    },
    "uncapping": {
        "common_params": {
            "damping": 200,
            "friction": 100,
            "density": 1.0,
            "mu": 3.0,
            "euler": [0.5 * np.pi, 0, np.pi]
        },
        "shape_setting": {
            "101685": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "101722": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "101727": [0.6, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "101732": [0.6, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "101736": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "101786": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "101787": [0.6, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "101793": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "101796": [0.6, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "102909": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "102910": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "102931": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "102942": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "102961": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
            "102970": [0.8, 3.0, "origin", 'closed', 0.25, 0.2, 0, 0.25, 0.2, 0],
        }
    }
}
shape_data = {}
for primact in src_cate_setting:
    common_params = src_cate_setting[primact]["common_params"]
    for shape_id, values in src_cate_setting[primact]["shape_setting"].items():

        shape_data[shape_id] = {field: value for field, value in zip(cate_field_name, values)}
        shape_data[shape_id].update(common_params)

for primact in trg_cate_setting:
    common_params = trg_cate_setting[primact]["common_params"]
    for shape_id, values in trg_cate_setting[primact]["shape_setting"].items():

        shape_data[shape_id] = {field: value for field, value in zip(cate_field_name, values)}
        shape_data[shape_id].update(common_params)
# Step 1: 写入数据到 JSON 文件
with open(cs_file, 'w') as f:
    json.dump(shape_data, f, indent=4)

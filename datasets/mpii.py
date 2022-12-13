import json
import os.path as path

from models.openpose import preprocess_image
from utils.misc import (
    compute_padded_coordinates,
    listdir_filtered,
    prepare_padded_video_data,
    prepare_ragged_data,
)

# Resources consulted on PCKh evaluation:
# http://human-pose.mpi-inf.mpg.de/#evaluation
# https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Andriluka_2D_Human_Pose_2014_CVPR_paper.pdf
# https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Pishchulin_DeepCut_Joint_Subset_CVPR_2016_paper.pdf

# Resources consulted on PCK evaluation:
# https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Andriluka_2D_Human_Pose_2014_CVPR_paper.pdf
# https://www.cs.cmu.edu/~deva/papers/pose_pami.pdf


JOINT_NAMES = [
    "right ankle",
    "right knee",
    "right hip",
    "left hip",
    "left knee",
    "left ankle",
    "pelvis",
    "thorax",
    "upper neck",
    "head top",
    "right wrist",
    "right elbow",
    "right shoulder",
    "left shoulder",
    "left elbow",
    "left wrist",
]

MAJOR_CAMERA_MOTION_TRAIN = [
    "001971054",
    "002058449",
    "002541913",
    "002651131",
    "004600264",
    "005808361",
    "009310277",
    "011483682",
    "011816483",
    "012758030",
    "016122129",
    "016261694",
    "017320187",
    "019068772",
    "022879817",
    "023107480",
    "025824439",
    "029214465",
    "029314773",
    "031172207",
    "034080354",
    "034106635",
    "035609673",
    "037461261",
    "037969241",
    "055204718",
    "055417800",
    "055680126",
    "058991769",
    "062894285",
    "066503667",
    "070723863",
    "071459592",
    "074543947",
    "076245869",
    "076949293",
    "081055333",
    "081379502",
    "081446848",
    "085841275",
    "086073058",
    "086606389",
    "088721274",
    "089255900",
    "090041806",
    "092969765",
    "098621074",
    "098727613",
    "099050514",
]

MAJOR_CAMERA_MOTION_TEST = [
    "001545662",
    "001658783",
    "002653220",
    "003594227",
    "003668320",
    "004315991",
    "004696597",
    "006936130",
    "010077800",
    "010312387",
    "011323805",
    "011355759",
    "011689804",
    "012063587",
    "012429722",
    "012466268",
    "013927207",
    "013935607",
    "014092422",
    "015253904",
    "015436272",
    "016162871",
    "016628337",
    "018143544",
    "019967378",
    "020788626",
    "022925140",
    "024014562",
    "024247050",
    "024734810",
    "025243694",
    "025794163",
    "025877998",
    "027993390",
    "028863055",
    "028975145",
    "029444889",
    "030137787",
    "032235384",
    "032295647",
    "032367924",
    "032875161",
    "033490561",
    "033799516",
    "033902979",
    "034825547",
    "034886827",
    "035597268",
    "036239390",
    "036891364",
    "037015870",
    "037064215",
    "037201809",
    "037454012",
    "037558295",
    "038089086",
    "038313853",
    "038389122",
    "038867729",
    "039785560",
    "041039683",
    "041877052",
    "042142455",
    "042413067",
    "042543644",
    "042710779",
    "043913491",
    "045054414",
    "045328883",
    "045825289",
    "046077877",
    "046369840",
    "047347177",
    "049161588",
    "050272039",
    "050313626",
    "051167744",
    "052224326",
    "052273283",
    "052439027",
    "053211349",
    "053261727",
    "053346109",
    "054355286",
    "055520081",
    "057343789",
    "057628836",
    "058058835",
    "058123822",
    "058304156",
    "058684112",
    "059089294",
    "059413072",
    "059865848",
    "060642906",
    "060671505",
    "062007276",
    "062423800",
    "062770805",
    "062939864",
    "064203165",
    "064591724",
    "066252228",
    "066519722",
    "067157320",
    "067756436",
    "070365786",
    "073086748",
    "073199394",
    "073644424",
    "074747094",
    "074760938",
    "075061475",
    "075454287",
    "075784611",
    "077987555",
    "079402519",
    "080219698",
    "080683474",
    "080987445",
    "081659944",
    "081722220",
    "081801477",
    "083493024",
    "083889809",
    "084964739",
    "085546058",
    "087720054",
    "087733935",
    "089711026",
    "089957457",
    "090415869",
    "090647211",
    "090932040",
    "092084188",
    "092469362",
    "092952914",
    "093436701",
    "093834046",
    "093983212",
    "094146481",
    "095071431",
    "095392858",
    "095486528",
    "096502691",
    "098003258",
    "098440218",
    "098484120",
]

MINOR_CAMERA_MOTION_TRAIN = [
    "003118314",
    "005637550",
    "006355835",
    "009120610",
    "014008402",
    "017436643",
    "024185229",
    "024398964",
    "034695320",
    "035335223",
    "039850473",
    "040076306",
    "041481950",
    "051423444",
    "055385428",
    "064114763",
    "067910232",
    "074825969",
    "078961814",
    "080839735",
    "081910232",
    "086472238",
    "087146059",
    "087737307",
    "089482735",
    "089609130",
    "097178208",
]

MINOR_CAMERA_MOTION_TEST = [
    "000919705",
    "001309446",
    "001822183",
    "007455064",
    "008413996",
    "008870253",
    "009398194",
    "012706051",
    "018812116",
    "020633372",
    "027134903",
    "027997909",
    "032533033",
    "033089802",
    "034920957",
    "035883048",
    "036426986",
    "037694443",
    "038816993",
    "039244511",
    "041892545",
    "042818415",
    "049261809",
    "049836777",
    "050761038",
    "051891771",
    "052030770",
    "052306542",
    "052451139",
    "054269608",
    "063226429",
    "065998405",
    "071334959",
    "072595568",
    "078517596",
    "084182878",
    "088708642",
    "089000534",
    "089204656",
    "089267171",
    "091484590",
    "091486439",
    "092556851",
    "092831856",
    "096361998",
    "096665482",
]

NO_CAMERA_MOTION_TRAIN = [
    "001099583",
    "008134878",
    "008534746",
    "010147154",
    "012887972",
    "013752094",
    "016080805",
    "020642127",
    "022197216",
    "037047210",
    "045221794",
    "047396164",
    "055676289",
    "059507793",
    "061812824",
    "069189520",
    "080285345",
    "082873751",
    "083021986",
    "087665360",
    "094290970",
    "096107032",
    "096930254",
    "097585208",
]

NO_CAMERA_MOTION_TEST = [
    "002007120",
    "005630328",
    "005755876",
    "007404465",
    "008175903",
    "008534967",
    "009028977",
    "011203890",
    "011881792",
    "013977238",
    "023221270",
    "032004960",
    "032207917",
    "036019064",
    "036778313",
    "037427080",
    "038610264",
    "038920168",
    "045083937",
    "046253472",
    "046267979",
    "046866064",
    "046869316",
    "049211629",
    "049442030",
    "050219501",
    "052770111",
    "053577468",
    "054561679",
    "054952888",
    "057447311",
    "061825477",
    "063183721",
    "063230583",
    "063340085",
    "063352971",
    "064328822",
    "066353737",
    "070407977",
    "070685734",
    "072900717",
    "074112065",
    "076898742",
    "077127035",
    "081546143",
    "082260056",
    "083244264",
    "083530250",
    "083832870",
    "085300946",
    "087357862",
    "089552323",
    "089855419",
    "092087240",
    "092658763",
    "094476573",
    "095006197",
    "095060715",
    "096990985",
    "098237862",
    "099687425",
]


def load_data(data_dir, split, size=(288, 512), n_frames_filter=41, video_id_filter=None):
    if split.startswith("auto"):
        return _load_auto_data(data_dir, split, size, n_frames_filter, video_id_filter)
    elif split == "test":
        return _load_test_data(data_dir, size, n_frames_filter, video_id_filter)
    elif split == "train":
        return _load_train_data(data_dir, size, n_frames_filter, video_id_filter)
    else:
        raise ValueError("Invalid split name {}.".format(split))


def split_by_camera_motion(items, data_dir, split, n_frames_filter=41):
    base_dir = path.join(data_dir, split)
    with open(path.join(base_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    no_motion = []
    minor_motion = []
    major_motion = []
    unlabeled = []
    i = 0
    for item in json_data[: len(items)]:
        video_id = item["video_id"]
        frames_i = listdir_filtered(path.join(base_dir, "frames", video_id), ".jpg")
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        item = items[i]
        i += 1
        if video_id in NO_CAMERA_MOTION_TRAIN + NO_CAMERA_MOTION_TEST:
            no_motion.append(item)
        elif video_id in MINOR_CAMERA_MOTION_TRAIN + MINOR_CAMERA_MOTION_TEST:
            minor_motion.append(item)
        elif video_id in MAJOR_CAMERA_MOTION_TRAIN + MAJOR_CAMERA_MOTION_TEST:
            major_motion.append(item)
        else:
            unlabeled.append(item)
    return no_motion, minor_motion, major_motion, unlabeled


def _load_auto_data(data_dir, split, size, n_frames_filter, video_id_filter):
    base_dir = path.join(data_dir, split)
    with open(path.join(base_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    frames = []
    joints = []
    n_people = []
    for item in json_data:
        frames_i = listdir_filtered(path.join(base_dir, "frames", item["video_id"]), ".jpg")
        if (video_id_filter is not None) and (item["video_id"] not in video_id_filter):
            continue
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        frames.append(frames_i)

        size_before = item["height"], item["width"]
        joints_i = []
        n_joints = 0
        max_people = 0
        for people_t in item["people"]:
            joints_t = []
            for person in people_t:
                joints_t.append(
                    [
                        list(compute_padded_coordinates(*joint, size_before, size))
                        for joint in person
                    ]
                )
                n_joints = len(person)
            max_people = max(max_people, len(people_t))
            joints_i.append(joints_t)
        empty_person = [[float("nan"), float("nan")]] * n_joints
        joints_i_padded = []
        for joints_t in joints_i:
            joints_i_padded.append(joints_t + [empty_person] * (max_people - len(joints_t)))
        joints.append(joints_i_padded)

        n_people_i = []
        for person_t in item["people"]:
            n_people_i.append(len(person_t))
        n_people.append(n_people_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_image)

    # Dataset shape (1, n_frames, n_people, 16, 2)
    joints = prepare_ragged_data(joints, inner_shape=(len(JOINT_NAMES), 2))

    # Dataset shape (1, n_frames)
    n_people = prepare_ragged_data(n_people)

    return (frames, joints, n_people), n_items


def _load_test_data(data_dir, size, n_frames_filter, video_id_filter):
    base_dir = path.join(data_dir, "test")
    with open(path.join(base_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    frames = []
    for item in json_data:
        frames_i = listdir_filtered(path.join(base_dir, "frames", item["video_id"]), ".jpg")
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        if (video_id_filter is not None) and (item["video_id"] not in video_id_filter):
            continue
        frames.append(frames_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_image)

    return frames, n_items


def _load_train_data(data_dir, size, n_frames_filter, video_id_filter):
    base_dir = path.join(data_dir, "train")
    with open(path.join(base_dir, "data.json"), "r") as f:
        json_data = json.load(f)

    frames = []
    frame_masks = []
    joints = []
    heads = []
    for item in json_data:
        frames_i = listdir_filtered(path.join(base_dir, "frames", item["video_id"]), ".jpg")
        if n_frames_filter and len(frames_i) != n_frames_filter:
            continue
        if (video_id_filter is not None) and (item["video_id"] not in video_id_filter):
            continue
        frames.append(frames_i)

        keyframe = list(map(path.basename, frames_i)).index(item["frame"])
        frame_mask_i = [False] * len(frames_i)
        frame_mask_i[keyframe] = True
        frame_masks.append(frame_mask_i)

        size_before = item["height"], item["width"]
        joints_i = []
        for person in item["people"]:
            joints_i.append(
                [
                    list(compute_padded_coordinates(*joint, size_before, size))
                    for joint in person["joints"]
                ]
            )
        joints_i = [joints_i] * len(frames_i)
        joints.append(joints_i)

        heads_i = []
        for person in item["people"]:
            x_1, x_2, y_1, y_2 = person["head"]
            x_1, y_1 = compute_padded_coordinates(x_1, y_1, size_before, size)
            x_2, y_2 = compute_padded_coordinates(x_2, y_2, size_before, size)
            heads_i.append([x_1, x_2, y_1, y_2])
        heads_i = [heads_i] * len(frames_i)
        heads.append(heads_i)

    n_items = len(frames)

    # Dataset shape (1, n_frames, size[0], size[1], 3)
    frames = prepare_padded_video_data(frames, size, preprocess_func=preprocess_image)

    # Dataset shape (1, n_frames)
    frame_masks = prepare_ragged_data(frame_masks)

    # Dataset shape (1, n_frames, n_people, 16, 2)
    joints = prepare_ragged_data(joints, inner_shape=(len(JOINT_NAMES), 2))

    # Dataset shape (1, n_frames, n_people, 4)
    heads = prepare_ragged_data(heads, inner_shape=(4,))

    return (frames, frame_masks, joints, heads), n_items

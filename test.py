
from pathlib import Path
import numpy as np
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (make_matching_plot,
                          read_image)

def test_matching():
    match_threshold = 0.2
    sinkhorn_iterations = 20
    nms_radius = 4
    keypoint_threshold = 0.005
    max_keypoints = 1024
    superglue = 'indoor'
    resize_float = False
    resize = [640, 480]
    output_dir = 'result/'
    input_dir = 'assets/scannet_sample_images/'

    torch.set_grad_enabled(False)

    device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    name0 = 'scene0806_00_frame-000225.jpg'
    name1 = 'scene0806_00_frame-001095.jpg'
    stem0, stem1 = Path(name0).stem, Path(name1).stem
    viz_path = output_dir / '{}_{}'.format(stem0, stem1)

    rot0, rot1 = 0, 0

    image0, inp0, _ = read_image(
        input_dir / name0, device, resize, rot0, resize_float)
    image1, inp1, _ = read_image(
        input_dir / name1, device, resize, rot1, resize_float)
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    color = cm.jet(mconf)
    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0)),
    ]
    if rot0 != 0 or rot1 != 0:
        text.append('Rotation: {}:{}'.format(rot0, rot1))
    # Display extra parameter info.
    k_thresh = matching.superpoint.config['keypoint_threshold']
    m_thresh = matching.superglue.config['match_threshold']
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {}:{}'.format(stem0, stem1),
    ]
    make_matching_plot(
        image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
        text, viz_path, opencv_title = 'Matches', small_text = small_text)

    expect = {(319.0 , 15.0) : (380.0 , 13.0),
              (329.0 , 21.0) : (317.0 , 10.0),
              (483.0 , 43.0) : (246.0 , 21.0),
              (204.0 , 176.0) : (86.0 , 46.0),
              (213.0 , 178.0) : (94.0 , 52.0),
              (212.0 , 190.0) : (52.0 , 51.0),
              (241.0 , 190.0) : (61.0 , 48.0),
              (605.0 , 190.0) : (437.0 , 320.0),
              (230.0 , 201.0) : (45.0 , 63.0),
              (602.0 , 201.0) : (502.0 , 340.0),
              (417.0 , 221.0) : (396.0 , 276.0),
              (202.0 , 226.0) : (18.0 , 63.0),
              (338.0 , 271.0) : (304.0 , 230.0),
              (306.0 , 370.0) : (186.0 , 386.0),
              (268.0 , 389.0) : (126.0 , 394.0),
              (152.0 , 407.0) : (66.0 , 272.0)}

    assert len(mkpts0) == len(mkpts1)
    for i in range(len(mkpts0)):
        assert expect[tuple(mkpts0[i])] == tuple(mkpts1[i])
    print('Assertion successful!\nCheck "{}/" directory to see result image'.format(output_dir))

test_matching()
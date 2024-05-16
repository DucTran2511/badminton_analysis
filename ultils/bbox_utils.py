def center_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return x, y


def l2_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

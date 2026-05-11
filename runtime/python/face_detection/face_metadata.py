FACE_CLASS = "face"
FACE_COLOR = (80, 180, 255)


def get_face_class_num() -> int:
    return 1


def get_face_label(_: int = 0) -> str:
    return FACE_CLASS


def get_face_det_palette(_: int = 0) -> tuple[int, int, int]:
    return FACE_COLOR

from fastapi import HTTPException


def reject_legacy_label_registration():
    raise HTTPException(410, "Đăng ký Label đã ngừng sử dụng. Hãy upload model tại Model / TensorRT, build và áp dụng cho camera để tracking và vẽ SAM2.")

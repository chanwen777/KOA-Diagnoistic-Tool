import logging
import SimpleITK as sitk


def save_sitk_image(image, output_path):
    """
    用 SimpleITK 保存图像，与普通图像（PIL/cv2 等）区分。
    医学或普通格式均由 output_path 后缀决定，不做额外规则判断。
    支持医学格式（.nii.gz、.nrrd、.mha 等）及 SimpleITK 支持的其它格式。
    若后缀不被 SimpleITK 支持，由 SimpleITK 抛出异常并向上传播。

    参数:
        image: SimpleITK.Image 对象
        output_path: 输出文件路径（str 或 Path），后缀决定格式

    返回:
        True 表示保存成功

    异常:
        保存失败时记录错误后重新抛出 SimpleITK 的异常
    """
    path_str = str(output_path)
    try:
        sitk.WriteImage(image, path_str)
        return True
    except Exception as e:
        logging.error("保存图像失败 path=%s 错误: %s", path_str, e, exc_info=True)
        raise


def load_sitk_image(input_path):
    """
    用 SimpleITK 读取图像，与普通图像（PIL/cv2 等）区分。
    医学或普通格式均由 input_path 后缀决定，不做额外规则判断。
    支持医学格式（.nii.gz、.nrrd、.mha 等）及 SimpleITK 支持的其它格式。
    若输入路径无效或文件无效，由 SimpleITK 抛出异常并向上传播。

    参数:
        input_path: 输入文件路径（str 或 Path），后缀决定格式

    返回:
        SimpleITK.Image

    异常:
        读取失败时记录错误后重新抛出 SimpleITK 的异常
    """
    path_str = str(input_path)
    try:
        return sitk.ReadImage(path_str)
    except Exception as e:
        logging.error("读取图像失败 path=%s 错误: %s", path_str, e, exc_info=True)
        raise

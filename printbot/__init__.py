"""printbot — autonomous 3D-print inspection, classification and removal on Viam.

Package layout (each subpackage can be developed and tested independently):

    printbot.types        shared data contracts between modules (no hardware deps)
    printbot.config       settings loading (JSON, no secrets)
    printbot.perception   camera acquisition, plate segmentation, depth/point-cloud, YOLO
    printbot.inspection   print-quality classification and STL/G-code reference comparison
    printbot.robot        Viam connection + arm/gripper/motion wrappers (dry-run by default)
    printbot.app          task state machine, operator approval, event log
"""

__version__ = "0.1.0"

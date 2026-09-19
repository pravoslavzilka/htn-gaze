"""Run (or flash) a UVC pipeline on a Luxonis OAK-1 so it shows up as a normal USB camera.

  python oak_uvc.py run     # host-run: camera acts as UVC while this script is running
  python oak_uvc.py flash   # write pipeline to the camera's flash (standalone UVC, no host needed)
"""
import sys
import time

import depthai as dai


def build_pipeline(device: "dai.Device | None" = None) -> dai.Pipeline:
    board = dai.BoardConfig()
    board.uvc = dai.BoardConfig.UVC(1920, 1080)
    pipeline = dai.Pipeline(device) if device is not None else dai.Pipeline(False)
    pipeline.setBoardConfig(board)
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setFps(30)
    uvc = pipeline.create(dai.node.UVC)
    cam.video.link(uvc.input)
    return pipeline


def run() -> None:
    cfg = dai.Device.Config()
    cfg.board.uvc = dai.BoardConfig.UVC(1920, 1080)
    dev = dai.Device(cfg, dai.Device.getAllAvailableDevices()[0])
    pipeline = build_pipeline(dev)
    pipeline.start()
    print("UVC running on", dev.getDeviceName(), "usb:", dev.getUsbSpeed(), "- Ctrl+C to stop", flush=True)
    while pipeline.isRunning():
        time.sleep(1)


def flash() -> None:
    info = dai.Device.getAllAvailableDevices()[0]
    bl = dai.DeviceBootloader(info, allowFlashingBootloader=True)
    print("running bootloader version:", bl.getVersion(), flush=True)
    ok, msg = bl.flashBootloader(lambda p: print(f"\rflashing bootloader {p * 100:5.1f}%", end="", flush=True))
    print("\nbootloader:", ok, msg, flush=True)
    if not ok:
        return
    ok, msg = bl.flash(lambda p: print(f"\rflashing pipeline {p * 100:5.1f}%", end="", flush=True), build_pipeline(),
                       applicationName="uvc")
    print("\npipeline:", ok, msg, flush=True)
    bl.close()


if __name__ == "__main__":
    {"run": run, "flash": flash}[sys.argv[1] if len(sys.argv) > 1 else "run"]()

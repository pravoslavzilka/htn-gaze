/* Minimal QNX 8 camera API header for on-target builds (libcamapi).
 * Layout matches QNX Camera library documentation used by the sample apps.
 */
#ifndef CAMERA_CAMERA_API_H
#define CAMERA_CAMERA_API_H

#include <stdint.h>
#include <errno.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CAMERA_EOK
#define CAMERA_EOK EOK
#endif

typedef int camera_error_t;

typedef enum {
    CAMERA_UNIT_NONE = 0,
    CAMERA_UNIT_1 = 1,
    CAMERA_UNIT_2 = 2,
    CAMERA_UNIT_3 = 3,
    CAMERA_UNIT_4 = 4,
    CAMERA_UNIT_NUM_UNITS = 16
} camera_unit_t;

enum {
    CAMERA_MODE_PREAD = 1 << 0,
    CAMERA_MODE_PWRITE = 1 << 1,
    CAMERA_MODE_DREAD = 1 << 2,
    CAMERA_MODE_DWRITE = 1 << 3,
    CAMERA_MODE_ROLL = 1 << 4,
    CAMERA_MODE_RO = (CAMERA_MODE_PREAD | CAMERA_MODE_DREAD),
    CAMERA_MODE_RW = (CAMERA_MODE_PREAD | CAMERA_MODE_PWRITE | CAMERA_MODE_DREAD | CAMERA_MODE_DWRITE)
};

typedef enum {
    CAMERA_FRAMETYPE_UNSPECIFIED = 0,
    CAMERA_FRAMETYPE_NV12,
    CAMERA_FRAMETYPE_RGB8888,
    CAMERA_FRAMETYPE_RGB888,
    CAMERA_FRAMETYPE_GRAY8,
    CAMERA_FRAMETYPE_BAYER,
    CAMERA_FRAMETYPE_CBYCRY,
    CAMERA_FRAMETYPE_COMPRESSEDVIDEO,
    CAMERA_FRAMETYPE_RGB565,
    CAMERA_FRAMETYPE_YCBCR420P,
    CAMERA_FRAMETYPE_YCBYCR,
    CAMERA_FRAMETYPE_YCRYCB,
    CAMERA_FRAMETYPE_CRYCBY,
    CAMERA_FRAMETYPE_ROI,
    CAMERA_FRAMETYPE_BAYER14_RGGB_PADLO16,
    CAMERA_FRAMETYPE_NV16,
    CAMERA_FRAMETYPE_NV24_Y12,
    CAMERA_FRAMETYPE_BGR8888
} camera_frametype_t;

typedef enum {
    CAMERA_STATUS_UNKNOWN = 0,
    CAMERA_STATUS_PHYSICAL_REMOVAL = 4,
    CAMERA_STATUS_BUFFERS_DETACHED = 5
} camera_devstatus_t;

typedef struct {
    uint32_t height;
    uint32_t width;
    uint32_t stride;
    int64_t uv_offset;
    int64_t uv_stride;
} camera_frame_nv12_t;

typedef struct {
    uint32_t height;
    uint32_t width;
    uint32_t stride;
} camera_frame_rgb8888_t;

typedef camera_frame_rgb8888_t camera_frame_bgr8888_t;
typedef camera_frame_rgb8888_t camera_frame_rgb888_t;

typedef union {
    camera_frame_nv12_t nv12;
    camera_frame_rgb8888_t rgb8888;
    camera_frame_rgb888_t rgb888;
    camera_frame_bgr8888_t bgr8888;
    uint8_t raw[64];
} camera_framedesc_t;

typedef struct {
    camera_frametype_t frametype;
    uint64_t framesize;
    uint8_t *framebuf;
    uint64_t framemetasize;
    void *framemeta;
    int64_t frametimestamp;
    camera_framedesc_t framedesc;
} camera_buffer_t;

typedef struct _camera_handle *camera_handle_t;
#define CAMERA_HANDLE_INVALID ((camera_handle_t)0)

typedef void (*camera_frame_callback_t)(camera_handle_t handle, camera_buffer_t *buffer, void *arg);
typedef void (*camera_status_callback_t)(camera_handle_t handle, camera_devstatus_t status,
                                         uint16_t extra, void *arg);

typedef enum {
    CAMERA_IMGPROP_FORMAT = 0,
    CAMERA_IMGPROP_WIDTH = 1,
    CAMERA_IMGPROP_HEIGHT = 2,
    CAMERA_IMGPROP_FRAMERATE = 3,
    CAMERA_IMGPROP_ROTATION = 4,
    CAMERA_IMGPROP_ZOOMFACTOR = 5,
    CAMERA_IMGPROP_HWOVERLAY = 6,
    CAMERA_IMGPROP_WIN_GROUPID = 7,
    CAMERA_IMGPROP_WIN_ID = 8,
    CAMERA_IMGPROP_CREATEWINDOW = 9
} camera_imgprop_t;

camera_error_t camera_open(camera_unit_t unit, uint32_t mode, camera_handle_t *handle);
camera_error_t camera_close(camera_handle_t handle);
camera_error_t camera_start_viewfinder(camera_handle_t handle, camera_frame_callback_t frame_cb,
                                       camera_status_callback_t status_cb, void *arg);
camera_error_t camera_stop_viewfinder(camera_handle_t handle);
camera_error_t camera_get_supported_vf_frame_types(camera_handle_t handle, uint32_t num_asked,
                                                   uint32_t *num, camera_frametype_t *types);
camera_error_t camera_private_get_vf_property(camera_handle_t handle, ...);
camera_error_t camera_private_set_vf_property(camera_handle_t handle, ...);

#define camera_get_vf_property(handle, ...) \
    camera_private_get_vf_property((handle), __VA_ARGS__, -1)
#define camera_set_vf_property(handle, ...) \
    camera_private_set_vf_property((handle), __VA_ARGS__, -1)

#ifdef __cplusplus
}
#endif

#endif
